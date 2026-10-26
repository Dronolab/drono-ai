import torch
from tqdm import tqdm


def train_model(
    model,
    dataloader,
    optimizer,
    loss_fn,
    epochs,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    history = []

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{epochs}",
        )

        for batch in progress_bar:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)

            out = model(images)

            loss = loss_fn(out, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        epoch_loss = total_loss / len(dataloader)
        history.append(epoch_loss)

        print(f"Epoch {epoch + 1} - Loss: {epoch_loss:.4f}")

    return model, history


def test_model(
    model,
    dataloader,
    loss_fn,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    total_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc="Testing",
    )

    with torch.no_grad():
        for batch in progress_bar:

            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)

            out = model(images)

            loss = loss_fn(out, images)

            total_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

    test_loss = total_loss / len(dataloader)

    print(f"Test Loss: {test_loss:.4f}")

    return test_loss


