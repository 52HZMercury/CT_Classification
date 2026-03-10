import yaml
import torch
import os
from tqdm import tqdm
from validation import validation


def train(Trainer, current_epoch, acc_best, auc_best, global_step_best, early_stop_counter,config):
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
        x, y = (batch["image"].to(Trainer.device), batch["label"].to(Trainer.device))

        Trainer.optimizer.zero_grad()
        logits = Trainer.model(x)

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
            if acc_val > acc_best:
                acc_best = acc_val
                auc_best = auc_val
                global_step_best = current_epoch
                early_stop_counter = 0

                checkpoint_dir = os.path.join(config['data']['out_dir'], f"{config['data']['exp_name']}/checkpoint")
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(Trainer.model.state_dict(), os.path.join(checkpoint_dir, f"{acc_best:.4f}_best_acc_model.pth"))

                # 更新打印信息
                print(f'New Best Acc: {acc_best:.4f}, AUC: {auc_val:.4f}, '
                      f'PPV: {ppv_val:.4f}, NPV: {npv_val:.4f}, '
                      f'Recall: {rec_val:.4f}, Specificity: {spec_val:.4f}')
            else:
                early_stop_counter += 1
                print(f'Not improved. Best Acc: {acc_best:.4f}, EarlyStop: {early_stop_counter}/{patience}')

            if early_stop_counter >= patience:
                stop_training = True

    return current_epoch, acc_best, auc_best, global_step_best, early_stop_counter, stop_training