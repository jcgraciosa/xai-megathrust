import torch


def predictions(model, val_loader, device):
    model.eval()
    predictions = []
    labels = []
    # Don't use gradients
    with torch.no_grad():
        # Loop through training data
        for (x_cont, x_region) in iter(
            val_loader
        ):  # Try to get the label here too. Then we can plot both the label and the prediction
            x_cont, x_region = x_cont.to(device), x_region.to(device)

            # Get outputs and calculate loss
            cat, _ = model(x_cont, x_region)
            _, pred_vals = torch.max(cat, 1)
            # _, label = torch.max(label, 1)
            predictions.append(pred_vals.item())
            # labels.append(label.item())
            # predictions = torch.cat((predictions, out), 0)
    # predictions = predictions.flatten()
    return predictions
