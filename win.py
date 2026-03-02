import sys
import compute
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QFileDialog,
    QLabel,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox
)
from PySide6.QtCore import QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure

import numpy as np
    
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)


class MainWindow(QMainWindow):
    solver : compute.ode_sys_solver
    dt = 0.1
    t = 0
    T_0 = np.array([100, 100, 100, 100, 100])
    def __init__(self):
        super().__init__()
        self.solver = compute.ode_sys_solver()
        self.setWindowTitle("ODE solver")
        self.resize(800, 600)
        self.label = QLabel("Время теплового расчета")

        file_menu = self.menuBar().addMenu("Файл")
        obj_open_action = file_menu.addAction("Открыть obj")
        obj_open_action.triggered.connect(self.open_obj)

        json_open_action = file_menu.addAction("Открыть json")
        json_open_action.triggered.connect(self.open_json)

        self.time_edge = QDoubleSpinBox()
        self.time_edge.setDecimals(2)
        self.time_edge.setValue(1.0)
        
        self.button_calc = QPushButton("Расчитать")
        self.button_inf_calc = QPushButton("Начать бесконечный расчет")
        self.button_stop_inf_calc = QPushButton("Остановить бесконечный расчет")
        self.button_calc.clicked.connect(self.calculate)
        self.button_inf_calc.clicked.connect(self.start_inf_calc)
        self.button_stop_inf_calc.clicked.connect(self.stop_inf_calc)


        self.canvas = MplCanvas()

        toolbar = NavigationToolbar2QT(self.canvas, self)

        self.timer = QTimer()
        self.timer.timeout.connect(self.inf_calc)

        layout = QVBoxLayout()
        layout.setSpacing(10)

        layout.addWidget(self.label)
        layout.addWidget(self.time_edge)
        layout.addWidget(self.button_calc)
        layout.addWidget(self.button_inf_calc) 
        layout.addWidget(self.button_stop_inf_calc)
        layout.addWidget(toolbar) 
        layout.addWidget(self.canvas)

        layout.addStretch()   # прижимает все к верху

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)


    def calculate(self):
        self.solver.T_0 = self.T_0
        self.solver.t_span = (0, self.time_edge.value())
        self.solver.solve_ode()
        self.canvas.ax.clear()
        for i in range(len(self.solver.sol.y)):
            self.canvas.ax.plot(self.solver.sol.t, self.solver.sol.y[i], label="T" + str(i + 1))

        self.canvas.draw()
        
    def start_inf_calc(self):
        self.solver.T_0 = self.T_0
        self.canvas.ax.clear()
        self.timer.start(500)

    def stop_inf_calc(self):
        self.timer.stop()

    def inf_calc(self):
        self.solver.t_span = (self.t, self.t + self.dt)
        self.solver.solve_ode()

        self.solver.T_0 = self.solver.sol.y[:, -1]
        self.t += self.dt
        
        for i in range(len(self.solver.sol.y)):
            self.canvas.ax.plot(self.solver.sol.t, self.solver.sol.y[i], label="T" + str(i + 1))

        self.canvas.draw()


    def open_obj(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать OBJ файл",
            "",
            "OBJ Files (*.obj)"
        )
        if file_path:
            self.solver.load_obj(file_path)

    def open_json(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать json файл",
            "",
            "json Files (*.json)"
        )
        self.solver.T_0 = np.array([100, 100, 100, 100, 100])
        if file_path:
            self.solver.load_json(file_path)
    
app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()