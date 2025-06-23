import json
import matplotlib.pyplot as plt

def plot_loss_train():
    # Đọc dữ liệu từ file training_history.json
    with open('training_history/training_history_2.json', 'r', encoding='utf-8') as f:
        history = json.load(f)

    # Lấy các epoch và loss
    epochs = [entry['epoch'] for entry in history]
    train_loss = [entry['train_loss'] for entry in history]
    val_loss = [entry['val_loss'] for entry in history]

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))  # Kích thước biểu đồ
    plt.plot(epochs, train_loss, label='Train Loss', color='red', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', color='blue', marker='o')

    # Thêm tiêu đề và nhãn
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Hiển thị biểu đồ
    plt.show()



if __name__ == "__main__":
    plot_loss_train()