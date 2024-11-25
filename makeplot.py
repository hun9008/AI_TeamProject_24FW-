import matplotlib.pyplot as plt
import re

# 여기서 경로 지정.
log_file = "./LeViT_FLA/LeViT_Flatten_log.txt"

epochs = []
train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []

with open(log_file, "r") as file:
    for line in file:
        epoch_match = re.match(r"Epoch (\d+)/\d+", line)
        if epoch_match:
            epochs.append(int(epoch_match.group(1)))

        train_match = re.match(r"Train Loss: ([\d\.]+), Train Accuracy: ([\d\.]+)%", line)
        if train_match:
            train_loss.append(float(train_match.group(1)))
            train_accuracy.append(float(train_match.group(2)))

        val_match = re.match(r"Validation Loss: ([\d\.]+), Validation Accuracy: ([\d\.]+)%", line)
        if val_match:
            val_loss.append(float(val_match.group(1)))
            val_accuracy.append(float(val_match.group(2)))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label="Train Loss", marker='o')
plt.plot(epochs, val_loss, label="Validation Loss", marker='o')
plt.title("Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, label="Train Accuracy", marker='o')
plt.plot(epochs, val_accuracy, label="Validation Accuracy", marker='o')
plt.title("Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()