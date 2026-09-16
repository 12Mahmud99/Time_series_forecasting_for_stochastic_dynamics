import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )
    
    val_loss=[]
    train_loss=[]
    epochs_list=[]
    last_avg_val=None
    count = 0
    patience = 10

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        epochs_list.append(epoch_no+1)
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                
                ##no effect if batch_size < itr_per_epoch
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()
            
        #store avg _loss
        train_loss.append(avg_loss/batch_no)
            
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
                        
            avg_valid = avg_loss_valid / batch_no
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best val loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
                ##saving loss    
                last_val_loss = avg_valid
                val_loss.append(avg_valid)
                count = 0 #reset when it improves
                print(f"\n best loss updated to {avg_loss_valid:.6f} at epoch {epoch_no}")
                torch.save(model.state_dict(), output_path)
            else:
                count = count +1
                if count >= patience:
                    print(f"\n Early Stopping {avg_loss_valid:.6f} invoked at epoch {epoch_no}")
                    break
                

        
    np.savez(
        config["training_curv"] + "/loss.npz",
        train_total=np.array(train_loss),
        val_total=np.array(val_loss),
        epochs=np.array(epochs_list),
    )
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_list, train_loss, 'b-', label='Train Loss')

    val_epochs = list(range(valid_epoch_interval, 
                        valid_epoch_interval * (len(val_loss) + 1), 
                        valid_epoch_interval))

    plt.plot(val_epochs, val_loss, 'r--', label='Val Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config['training_curv']}/loss_curves.png", dpi=150)


    '''if foldername != "":
        torch.save(model.state_dict(), output_path)'''


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )
            CRPS_sum = calc_quantile_CRPS_sum(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(
                foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)
                print("CRPS_sum:", CRPS_sum)


def evaluate_csdi_nips(model, test_loader, n_samples, device, prediction_length, context_length):
    model.eval()
    all_samples = []
    all_gt = []
    all_contexts = []
    time_elapsed = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Forecasting"):
            (observed_data, observed_mask, observed_tp,
             gt_mask, _, _, _) = model.process_data(batch)

            # observed_data: (B, K, L) where L = context_length + prediction_length
            # Extract the context part (first context_length steps)
            context_part = observed_data[:, :, :context_length]  # (B, K, context_length)

            # Compute per‑trajectory scale from the context ONLY
            scales = torch.mean(torch.abs(context_part), dim=2, keepdim=True)  # (B, 1, 1)
            scales = torch.clamp(scales, min=1e-8)

            # Normalize the entire observed_data (context + future) using the context scale
            observed_data_norm = observed_data / scales

            cond_mask = gt_mask
            side_info = model.get_side_info(observed_tp, cond_mask)

            start_time = timelib.time()
            samples_norm = model.impute(observed_data_norm, cond_mask, side_info, n_samples)
            time_elapsed += timelib.time() - start_time

            # samples_norm: (B, n_samples, K, L) -> extract K=0, future part
            samples_norm = samples_norm[:, :, 0, -prediction_length:]  # (B, n_samples, H)
            samples_norm = samples_norm.permute(0, 2, 1)               # (B, H, n_samples)

            # Denormalize the samples using the same context scale
            samples = samples_norm * scales.squeeze(-1).squeeze(-1)[:, None, None]

            # Ground truth and contexts (raw, from observed_data)
            gt = observed_data[:, 0, -prediction_length:].clone()      # (B, H)
            contexts = observed_data[:, 0, :context_length].clone()    # (B, L)

            all_samples.append(samples.cpu().numpy())
            all_gt.append(gt.cpu().numpy())
            all_contexts.append(contexts.cpu().numpy())

    samples_out = np.concatenate(all_samples, axis=0)
    gt_out = np.concatenate(all_gt, axis=0)
    contexts_out = np.concatenate(all_contexts, axis=0)
    return samples_out, gt_out, contexts_out, time_elapsed