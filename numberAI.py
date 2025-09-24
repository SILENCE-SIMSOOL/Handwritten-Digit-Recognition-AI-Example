import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ----------------- 초기 모델 -----------------
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(9, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = []
train_labels = []

# ----------------- 공통 캔버스 클래스 -----------------
class CanvasFrame(tk.Frame):
    def __init__(self, master, width=280, height=280):
        super().__init__(master)
        self.canvas = tk.Canvas(self, width=width, height=height, bg="white")
        self.canvas.pack()
        self.image = Image.new("L", (width,height), 255)
        self.draw_obj = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.draw)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
        self.draw_obj.ellipse([x-r, y-r, x+r, y+r], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.draw_obj.rectangle([0,0,self.image.width,self.image.height], fill=255)

# ----------------- 메인 애플리케이션 -----------------
class DigitApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("숫자 인식 AI")
        self.geometry("400x450")
        self.current_frame = None
        self.show_main_menu()

    # ---------- 화면 전환 ----------
    def switch_frame(self, new_frame_class):
        if self.current_frame is not None:
            self.current_frame.destroy()
        self.current_frame = new_frame_class(self)
        self.current_frame.pack()

    # ---------- 메인 메뉴 ----------
    def show_main_menu(self):
        self.switch_frame(MainMenu)

    # ---------- 학습 메뉴 ----------
    def show_train_menu(self):
        self.switch_frame(TrainMenu)

    # ---------- 예측 메뉴 ----------
    def show_predict_menu(self):
        self.switch_frame(PredictMenu)

# ---------- 메인 메뉴 프레임 ----------
class MainMenu(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        tk.Label(self, text="메인 메뉴", font=("Arial", 20)).pack(pady=20)
        tk.Button(self, text="학습 모드", width=20, command=master.show_train_menu).pack(pady=10)
        tk.Button(self, text="실전 모드", width=20, command=master.show_predict_menu).pack(pady=10)

# ---------- 학습 메뉴 프레임 ----------
class TrainMenu(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        tk.Label(self, text="학습 모드", font=("Arial", 16)).pack(pady=10)
        self.canvas_frame = CanvasFrame(self)
        self.canvas_frame.pack()

        # 숫자 선택 라디오 버튼
        self.selected_number = tk.IntVar()
        self.selected_number.set(1)
        radio_frame = tk.Frame(self)
        radio_frame.pack()
        for i in range(1,10):
            tk.Radiobutton(radio_frame, text=str(i), variable=self.selected_number, value=i).pack(side=tk.LEFT)

        # 버튼
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="제출", command=self.submit).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="초기화", command=self.canvas_frame.clear).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="메인으로", command=master.show_main_menu).pack(side=tk.LEFT, padx=5)

    def submit(self):
        img = self.canvas_frame.image.resize((28,28)).convert("L")
        arr = np.array(img).astype("float32") / 255.0
        arr = arr.reshape(784,)
        train_images.append(arr)
        label = self.selected_number.get() - 1
        train_labels.append(label)
        # 모델 학습
        x_train = np.array(train_images)
        y_train = keras.utils.to_categorical(np.array(train_labels), 9)
        model.fit(x_train, y_train, epochs=1, batch_size=8, verbose=0)
        self.canvas_frame.clear()
        print(f"제출 완료! 총 학습 샘플: {len(train_images)}")

# ---------- 예측 메뉴 프레임 ----------
class PredictMenu(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        tk.Label(self, text="실전 모드", font=("Arial", 16)).pack(pady=10)
        self.canvas_frame = CanvasFrame(self)
        self.canvas_frame.pack()

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="예측", command=self.predict).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="초기화", command=self.canvas_frame.clear).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="메인으로", command=master.show_main_menu).pack(side=tk.LEFT, padx=5)

        self.result_label = tk.Label(self, text="예측 결과")
        self.result_label.pack()
        self.prob_label = tk.Label(self, text="")
        self.prob_label.pack()

    def predict(self):
        img = self.canvas_frame.image.resize((28,28)).convert("L")
        arr = np.array(img).astype("float32") / 255.0
        arr = arr.reshape(1,784)
        pred = model.predict(arr)
        number = np.argmax(pred) + 1
        probs = [f"{i+1}: {pred[0,i]*100:.1f}%" for i in range(9)]
        self.result_label.config(text=f"예측 숫자: {number}")
        self.prob_label.config(text="\n".join(probs))

# ---------- 실행 ----------
if __name__ == "__main__":
    app = DigitApp()
    app.mainloop()
