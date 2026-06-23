import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tool.Trainer import Trainer
from tool.dataset.mnist import load_mnist


def accuracy(x, t):
    y_pred = np.argmax(trainer.predict(x), axis=1)
    y = np.argmax(t, axis=1)
    return np.mean(y_pred == y)


def show_image(image, label, pred_label=None):
    plt.imshow(image.reshape(28, 28), cmap='gray')
    true_label = np.argmax(label)
    title = f'y: {true_label}' + (f', pre_y: {pred_label}' if pred_label is not None else '')
    plt.title(title)
    plt.axis('off')
    plt.show()


def test_choose():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    choose = int(input(f'请输入要测试的样本编号(0-{x_test.shape[0]-1})：'))
    y_pred = np.argmax(trainer.predict(x_test[choose].reshape(1, -1)))
    show_image(x_test[choose], t_test[choose], y_pred)


def preprocess_image(image):
    from PIL import Image

    img_array = image.copy()
    non_zero = np.where(img_array > 0)
    
    if len(non_zero[0]) == 0:
        return np.zeros(784)
    
    min_row, max_row = np.min(non_zero[0]), np.max(non_zero[0])
    min_col, max_col = np.min(non_zero[1]), np.max(non_zero[1])
    
    padding = 20
    min_row, max_row = max(0, min_row - padding), min(img_array.shape[0]-1, max_row + padding)
    min_col, max_col = max(0, min_col - padding), min(img_array.shape[1]-1, max_col + padding)
    
    img_cropped = img_array[min_row:max_row+1, min_col:max_col+1]
    
    max_dim = max(img_cropped.shape)
    padded = np.zeros((max_dim, max_dim), dtype=np.uint8)
    offset_row, offset_col = (max_dim - img_cropped.shape[0]) // 2, (max_dim - img_cropped.shape[1]) // 2
    padded[offset_row:offset_row+img_cropped.shape[0], offset_col:offset_col+img_cropped.shape[1]] = img_cropped
    
    img_resized = Image.fromarray(padded).resize((20, 20), Image.LANCZOS)
    img_resized_array = np.array(img_resized)
    img_resized_array = np.where(img_resized_array > 50, 255, 0).astype(np.uint8)
    
    centered = np.zeros((28, 28), dtype=np.uint8)
    centered[4:24, 4:24] = img_resized_array
    
    return centered.flatten() / 255.0


class DrawingCanvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(8, 6))
        
        self.ax = self.fig.add_axes([0.1, 0.2, 0.8, 0.7])
        self.ax.set_title('Draw a digit (drag mouse)')
        self.ax.set_xlim(0, 280)
        self.ax.set_ylim(0, 280)
        self.ax.invert_yaxis()
        self.ax.grid(True)
        
        self.canvas = np.zeros((280, 280), dtype=np.uint8)
        self.drawing = False
        self.last_pos = None
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        self.btn_predict = Button(self.fig.add_axes([0.25, 0.05, 0.2, 0.1]), 'Predict', color='lightblue', hovercolor='skyblue')
        self.btn_clear = Button(self.fig.add_axes([0.55, 0.05, 0.2, 0.1]), 'Clear', color='lightgray', hovercolor='gray')
        self.btn_predict.on_clicked(self.predict_digit)
        self.btn_clear.on_clicked(self.clear_canvas)
        
        self.imshow = self.ax.imshow(self.canvas, cmap='gray', vmin=0, vmax=255)
        plt.show()
    
    def draw_pixel(self, x, y):
        for i in range(-5, 6):
            for j in range(-5, 6):
                nx, ny = x + i, y + j
                if 0 <= nx < 280 and 0 <= ny < 280:
                    self.canvas[ny, nx] = 255
    
    def on_press(self, event):
        if event.inaxes == self.ax and event.button == 1:
            self.drawing = True
            self.last_pos = (int(event.xdata), int(event.ydata))
            self.draw_pixel(*self.last_pos)
    
    def on_release(self, event):
        if event.button == 1:
            self.drawing = False
            self.last_pos = None
    
    def on_motion(self, event):
        if self.drawing and event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)
            if self.last_pos is not None:
                dx, dy = x - self.last_pos[0], y - self.last_pos[1]
                steps = max(abs(dx), abs(dy), 1)
                for s in range(steps + 1):
                    sx = int(self.last_pos[0] + dx * s / steps)
                    sy = int(self.last_pos[1] + dy * s / steps)
                    self.draw_pixel(sx, sy)
            self.last_pos = (x, y)
            self.imshow.set_data(self.canvas)
            self.fig.canvas.draw()
    
    def clear_canvas(self, event):
        self.canvas.fill(0)
        self.imshow.set_data(self.canvas)
        self.fig.canvas.draw()
    
    def predict_digit(self, event):
        processed = preprocess_image(self.canvas)
        y_pred = trainer.predict(processed.reshape(1, -1))
        pred_label = np.argmax(y_pred)
        confidence = np.max(y_pred)
        
        plt.figure(figsize=(8, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(self.canvas, cmap='gray')
        plt.title('Your drawing')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(processed.reshape(28, 28), cmap='gray')
        plt.title('Processed image')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        bars = plt.bar(range(10), y_pred[0])
        bars[pred_label].set_color('red')
        plt.title(f'Prediction: {pred_label}\nConfidence: {confidence:.2f}')
        plt.xticks(range(10))
        
        plt.tight_layout()
        plt.show()


def test_drawing():
    print('Draw a digit on the canvas and click Predict')
    DrawingCanvas()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.load_model('model.op')
    trainer.show_network()
    
    while True:
        print('\n请选择测试方式：')
        print('1. 测试集准确率')
        print('2. 手动选择MNIST图片')
        print('3. 鼠标绘画预测')
        print('4. 退出')
        choice = input('请输入选项(1-4)：')
        
        if choice == '1':
            (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
            print(f'\n测试集准确率：{accuracy(x_test, t_test)*100:.2f}%')
        elif choice == '2':
            test_choose()
        elif choice == '3':
            test_drawing()
        elif choice == '4':
            break
        else:
            print('无效选项，请重新输入')
