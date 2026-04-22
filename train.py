import yaml
import torch
import os
from tqdm import tqdm
from validation import validation
import glob  # 新增：用于查找旧的权重文件

def train(Trainer, current_epoch, acc_best, auc_best, ppv_best, npv_best, rec_best,spec_best, global_step_best, early_stop_counter,config):
    """
    针对分类任务修改的训练流程
    """
    Trainer.model.train()
    epoch_loss = 0
    num_steps = len(Trainer.train_loader)
    stop_training = False
    patience = config['training']['patience']

    epoch_iterator = tqdm(Trainer.train_loader, desc=f"Epoch {current_epoch} Training", dynamic_ncols=True)

    for step, batch in enumerate(epoch_iterator):
        # 1. 准备图像数据 [B, 3, H, W, D]
        img = batch["image"].to(Trainer.device)
        # ca = batch["CA"].to(Trainer.device)
        cac = batch["CAC"].to(Trainer.device)
        x_image = torch.cat([img, cac], dim=1)

        # pred = batch["pred"].to(Trainer.device)
        # x_image = torch.cat([img, pred], dim=1)

        # 2. 准备临床数据 [B, Feature_Dim]
        # 确保从 batch 拿出来的是 Tensor，并转为 float32
        x_tabular = batch["tabular_features"]
        if isinstance(x_tabular, list):
            x_tabular = torch.stack(x_tabular, dim=1)  # 处理可能的格式转换
        x_tabular = x_tabular.to(Trainer.device).float()

        # 3. 准备标签
        y = batch["label"].to(Trainer.device)

        # 4. 前向传播：传入两个输入
        Trainer.optimizer.zero_grad()

        # 【关键修改】：模型现在接收两个变量
        logits = Trainer.model(x_image, x_tabular)
        # logits = Trainer.model(x_image)

        loss = Trainer.loss_function(logits, y.long())

        loss.backward()
        Trainer.optimizer.step()

        epoch_loss += loss.item()
        epoch_iterator.set_description(f"Epoch {current_epoch} Loss: {loss.item():.4f}")
        Trainer.writer.add_scalar('Train/Loss', loss.item(), Trainer.global_step)
        Trainer.global_step += 1

        # Epoch 结束后的验证
        if step + 1 == num_steps:
            # 调用分类版的 validation (下方会定义)
            acc_val, auc_val, ppv_val, npv_val, rec_val, spec_val = validation(Trainer)

            epoch_loss /= num_steps
            Trainer.scheduler.step(acc_val)

            # 2. 写入 TensorBoard
            Trainer.writer.add_scalar('Validation/Accuracy', acc_val, current_epoch)
            Trainer.writer.add_scalar('Validation/AUC', auc_val, current_epoch)
            Trainer.writer.add_scalar('Validation/PPV(Precision)', ppv_val, current_epoch)
            Trainer.writer.add_scalar('Validation/NPV', npv_val, current_epoch)
            Trainer.writer.add_scalar('Validation/Recall', rec_val, current_epoch)
            Trainer.writer.add_scalar('Validation/Specificity', spec_val, current_epoch)

            # 3. 顺便可以在控制台打印出这些指标
            # 监控auc指标
            if auc_val > auc_best:
                acc_best = acc_val
                auc_best = auc_val
                ppv_best = ppv_val
                npv_best = npv_val
                rec_best = rec_val
                spec_best = spec_val
                global_step_best = current_epoch
                early_stop_counter = 0

                # checkpoint_dir = os.path.join(config['data']['out_dir'], f"{config['data']['exp_name']}/checkpoint")
                # os.makedirs(checkpoint_dir, exist_ok=True)
                # torch.save(Trainer.model.state_dict(), os.path.join(checkpoint_dir, f"{auc_best:.4f}_best_auc_model.pth"))

                # 保存模型
                checkpoint_dir = os.path.join(config['data']['out_dir'], f"{config['data']['exp_name']}/checkpoint")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                else:
                    # 【修改点】查找并删除该目录下以前保存的 best_metric_model 权重
                    old_checkpoints = glob.glob(os.path.join(checkpoint_dir, "best_metric_model_*.pth"))
                    for old_ckpt in old_checkpoints:
                        try:
                            os.remove(old_ckpt)
                        except OSError:
                            pass

                # 保存新的权重文件
                torch.save(Trainer.model.state_dict(),
                           os.path.join(checkpoint_dir, f"best_metric_model_{auc_val:.4f}.pth"))
                # 更新打印信息
                print(f'New Best Acc: {acc_best:.4f}, AUC: {auc_val:.4f}, '
                      f'PPV: {ppv_val:.4f}, NPV: {npv_val:.4f}, '
                      f'Recall: {rec_val:.4f}, Specificity: {spec_val:.4f}')
            else:
                early_stop_counter += 1
                print(f'Not improved. Best Acc: {acc_best:.4f}, EarlyStop: {early_stop_counter}/{patience}')

            if early_stop_counter >= patience:
                stop_training = True

    return current_epoch, acc_best, auc_best, ppv_best, npv_best, rec_best, spec_best, global_step_best, early_stop_counter, stop_training