import logging
import os.path as osp
import shutil
import time

import numpy as np
import torch
from progress.bar import Bar

from lib.core.config import VIBE_DATA_DIR
from lib.utils.eval_utils import compute_accel, compute_error_accel, \
    compute_error_verts, batch_compute_similarity_transform_torch, compute_pck
from lib.utils.utils import move_dict_to_device, AverageMeter


def validate(model, device, test_loader):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    logger = logging.getLogger(__name__)
    start = time.time()

    summary_string = ''

    bar = Bar('Validation', fill='#', max=len(test_loader))

    evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts'])
    for k, v in evaluation_accumulators.items():
        evaluation_accumulators[k] = []

    J_regressor = torch.from_numpy(np.load(osp.join(VIBE_DATA_DIR, 'J_regressor_h36m.npy'))).float()

    for i, target in enumerate(test_loader):
        move_dict_to_device(target, device)

        # <=============
        with torch.no_grad():
            inp = target['features']

            preds = model(inp, J_regressor=J_regressor)

            # convert to 14 keypoint format for evaluation
            n_kp = preds[-1]['kp_3d'].shape[-2]
            pred_j3d = preds[-1]['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
            target_j3d = target['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
            pred_verts = preds[-1]['verts'].view(-1, 6890, 3).cpu().numpy()
            target_theta = target['theta'].view(-1, 85).cpu().numpy()

            evaluation_accumulators['pred_verts'].append(pred_verts)
            evaluation_accumulators['target_theta'].append(target_theta)

            evaluation_accumulators['pred_j3d'].append(pred_j3d)
            evaluation_accumulators['target_j3d'].append(target_j3d)
        # =============>

        batch_time = time.time() - start

        summary_string = f'({i + 1}/{len(test_loader)}) | batch: {batch_time * 10.0:.4}ms | ' \
                         f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'

        bar.suffix = summary_string
        bar.next()

    bar.finish()

    logger.info(summary_string)

    return evaluation_accumulators


def evaluate(evaluation_accumulators, dataset='ThreeDPW'):
    for k, v in evaluation_accumulators.items():
        evaluation_accumulators[k] = np.vstack(v)

    pred_j3ds = evaluation_accumulators['pred_j3d']
    target_j3ds = evaluation_accumulators['target_j3d']

    pred_j3ds = torch.from_numpy(pred_j3ds).float()
    target_j3ds = torch.from_numpy(target_j3ds).float()

    print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')

    pred_pelvis = (pred_j3ds[:, [2], :] + pred_j3ds[:, [3], :]) / 2.0
    target_pelvis = (target_j3ds[:, [2], :] + target_j3ds[:, [3], :]) / 2.0

    pred_j3ds -= pred_pelvis
    target_j3ds -= target_pelvis

    # Absolute error (MPJPE)
    errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
    errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

    pred_verts = evaluation_accumulators['pred_verts']
    target_theta = evaluation_accumulators['target_theta']

    m2mm = 1000

    pve = np.mean(compute_error_verts(target_theta=target_theta, pred_verts=pred_verts)) * m2mm
    accel = np.mean(compute_accel(pred_j3ds)) * m2mm
    accel_err = np.mean(compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)) * m2mm
    mpjpe = np.mean(errors) * m2mm
    pa_mpjpe = np.mean(errors_pa) * m2mm
    if dataset == 'MPII3D':
        pck = compute_pck(pred_j3ds, target_j3ds)
    else:
        pck = [-1]

    eval_dict = {
        'mpjpe': mpjpe,
        'pa-mpjpe': pa_mpjpe,
        'pve': pve,
        'accel': accel,
        'accel_err': accel_err,
        'pck_mean': pck[-1],
    }

    log_str = ' '.join([f'{k.upper()}: {v:.4f},' for k, v in eval_dict.items()])
    print(log_str)

    return pa_mpjpe


def train(data_loaders, generator, motion_discriminator, gen_optimizer, dis_motion_optimizer, dis_motion_update_steps,
          end_epoch, criterion, start_epoch=0, lr_scheduler=None, motion_lr_scheduler=None, device=None,
          writer=None, logdir='output', resume=None, performance_type='min',
          num_iters_per_epoch=1000,
          ):
    logger = logging.getLogger(__name__)
    # Prepare dataloaders
    train_2d_loader, train_3d_loader, disc_motion_loader, valid_loader = data_loaders
    disc_motion_iter = iter(disc_motion_loader)

    train_2d_iter = train_3d_iter = None
    if train_2d_loader:
        train_2d_iter = iter(train_2d_loader)
    if train_3d_loader:
        train_3d_iter = iter(train_3d_loader)

    best_performance = float('inf') if performance_type == 'min' else -float('inf')
    train_global_step = 0

    if writer is None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=logdir)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Resume from a pretrained model
    if resume is not None:
        resume_pretrained(resume, generator, gen_optimizer, motion_discriminator, dis_motion_optimizer, logger)

    # Fit
    for epoch in range(start_epoch, end_epoch):
        train_global_step = train_one_epoch(
            criterion, device, dis_motion_optimizer, dis_motion_update_steps, disc_motion_iter,
            disc_motion_loader, end_epoch, epoch, gen_optimizer, generator, logger, motion_discriminator,
            num_iters_per_epoch, train_2d_iter, train_2d_loader, train_3d_iter, train_3d_loader,
            train_global_step, writer)

        evaluation_accumulators = validate(generator, device, valid_loader)
        performance = evaluate(evaluation_accumulators)

        if lr_scheduler is not None:
            lr_scheduler.step(performance)

        if motion_lr_scheduler is not None:
            motion_lr_scheduler.step(performance)

        # log the learning rate
        for param_group in gen_optimizer.param_groups:
            print(f'Learning rate {param_group["lr"]}')
            writer.add_scalar('lr/gen_lr', param_group['lr'], global_step=epoch)

        for param_group in dis_motion_optimizer.param_groups:
            print(f'Learning rate {param_group["lr"]}')
            writer.add_scalar('lr/dis_lr', param_group['lr'], global_step=epoch)

        logger.info(f'Epoch {epoch + 1} performance: {performance:.4f}')

        if performance_type == 'min':
            is_best = performance < best_performance
        else:
            is_best = performance > best_performance

        if is_best:
            best_performance = performance
        save_model(performance, epoch, generator, gen_optimizer, motion_discriminator,
                   dis_motion_optimizer, is_best, logdir, logger)

    writer.close()


def train_one_epoch(criterion, device, dis_motion_optimizer, dis_motion_update_steps, disc_motion_iter,
                    disc_motion_loader, end_epoch, epoch, gen_optimizer, generator, logger, motion_discriminator,
                    num_iters_per_epoch, train_2d_iter, train_2d_loader, train_3d_iter, train_3d_loader,
                    train_global_step, writer):
    # Single epoch training routine
    losses = AverageMeter()
    timer = {
        'data': 0,
        'forward': 0,
        'loss': 0,
        'backward': 0,
        'batch': 0,
    }
    generator.train()
    motion_discriminator.train()
    start = time.time()
    summary_string = ''
    bar = Bar(f'Epoch {epoch + 1}/{end_epoch}', fill='#', max=num_iters_per_epoch)
    for i in range(num_iters_per_epoch):
        # Dirty solution to reset an iterator
        target_2d = target_3d = None
        if train_2d_iter:
            try:
                target_2d = next(train_2d_iter)
            except StopIteration:
                train_2d_iter = iter(train_2d_loader)
                target_2d = next(train_2d_iter)

            move_dict_to_device(target_2d, device)

        if train_3d_iter:
            try:
                target_3d = next(train_3d_iter)
            except StopIteration:
                train_3d_iter = iter(train_3d_loader)
                target_3d = next(train_3d_iter)

            move_dict_to_device(target_3d, device)

        real_body_samples = real_motion_samples = None

        try:
            real_motion_samples = next(disc_motion_iter)
        except StopIteration:
            disc_motion_iter = iter(disc_motion_loader)
            real_motion_samples = next(disc_motion_iter)

        move_dict_to_device(real_motion_samples, device)

        # <======= Feedforward generator and discriminator
        if target_2d and target_3d:
            inp = torch.cat((target_2d['features'], target_3d['features']), dim=0).to(device)
        elif target_3d:
            inp = target_3d['features'].to(device)
        else:
            inp = target_2d['features'].to(device)

        timer['data'] = time.time() - start
        start = time.time()

        preds = generator(inp)

        timer['forward'] = time.time() - start
        start = time.time()

        gen_loss, motion_dis_loss, loss_dict = criterion(
            generator_outputs=preds,
            data_2d=target_2d,
            data_3d=target_3d,
            data_body_mosh=real_body_samples,
            data_motion_mosh=real_motion_samples,
            motion_discriminator=motion_discriminator,
        )
        # =======>

        timer['loss'] = time.time() - start
        start = time.time()

        # <======= Backprop generator and discriminator
        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()

        if train_global_step % dis_motion_update_steps == 0:
            dis_motion_optimizer.zero_grad()
            motion_dis_loss.backward()
            dis_motion_optimizer.step()
        # =======>

        # <======= Log training info
        total_loss = gen_loss + motion_dis_loss

        losses.update(total_loss.item(), inp.size(0))

        timer['backward'] = time.time() - start
        timer['batch'] = timer['data'] + timer['forward'] + timer['loss'] + timer['backward']
        start = time.time()

        summary_string = f'({i + 1}/{num_iters_per_epoch}) | Total: {bar.elapsed_td} | ' \
                         f'ETA: {bar.eta_td:} | loss: {losses.avg:.4f}'

        for k, v in loss_dict.items():
            summary_string += f' | {k}: {v:.2f}'
            writer.add_scalar('train_loss/' + k, v, global_step=train_global_step)

        for k, v in timer.items():
            summary_string += f' | {k}: {v:.2f}'

        writer.add_scalar('train_loss/loss', total_loss.item(), global_step=train_global_step)

        train_global_step += 1
        bar.suffix = summary_string
        bar.next()

        if torch.isnan(total_loss):
            exit('Nan value in loss, exiting!...')
        # =======>

    bar.finish()
    logger.info(summary_string)

    return train_global_step


def save_model(performance, epoch, generator, gen_optimizer, motion_discriminator,
               dis_motion_optimizer, is_best, logdir, logger):
    save_dict = {
        'epoch': epoch,
        'gen_state_dict': generator.state_dict(),
        'performance': performance,
        'gen_optimizer': gen_optimizer.state_dict(),
        'disc_motion_state_dict': motion_discriminator.state_dict(),
        'disc_motion_optimizer': dis_motion_optimizer.state_dict(),
    }

    filename = osp.join(logdir, 'checkpoint.pth.tar')
    torch.save(save_dict, filename)

    if is_best:
        logger.info('Best performance achived, saving it!')
        shutil.copyfile(filename, osp.join(logdir, 'model_best.pth.tar'))

        with open(osp.join(logdir, 'best.txt'), 'w') as f:
            f.write(str(float(performance)))


def resume_pretrained(model_path, generator, gen_optimizer, motion_discriminator, dis_motion_optimizer, logger):
    if osp.isfile(model_path):
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        generator.load_state_dict(checkpoint['gen_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        best_performance = checkpoint['performance']

        if 'disc_motion_optimizer' in checkpoint.keys():
            motion_discriminator.load_state_dict(checkpoint['disc_motion_state_dict'])
            dis_motion_optimizer.load_state_dict(checkpoint['disc_motion_optimizer'])

        logger.info(f"=> loaded checkpoint '{model_path}' "
                    f"(epoch {start_epoch}, performance {best_performance})")
    else:
        logger.info(f"=> no checkpoint found at '{model_path}'")
