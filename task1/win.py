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
    QDoubleSpinBox,
    QToolBox,
    QMenu,
    QToolBar
)
from PySide6.QtGui import QAction
from PySide6.QtCore import QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure

import numpy as np

from PySide6.QtGui import QAction, QIcon
    
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)


class MainWindow(QMainWindow):
    solver : compute.ode_sys_solver
    dt = 1
    t = 0
    obj_loaded = False
    json_loaded = False
    solution_y : list()
    solution_x : list()
    def __init__(self):
        super().__init__()
        self.solver = compute.ode_sys_solver()
        self.setWindowTitle("ODE solver")
        self.resize(1280, 720)
        self.label = QLabel("Время теплового расчета")
        toolbar = QToolBar("My Toolbar")
        self.addToolBar(toolbar)

        open_obj_action = QAction("Открыть obj", self)
        open_obj_action.triggered.connect(self.open_obj)
        toolbar.addAction(open_obj_action)

        open_json_action = QAction("Открыть json", self)
        open_json_action.triggered.connect(self.open_json)
        toolbar.addAction(open_json_action)

        calc_action = QAction("Расчитать", self)
        calc_action.triggered.connect(self.calculate)
        toolbar.addAction(calc_action)

        start_inf_calc_action = QAction("Начать бесконечный расчет", self)
        start_inf_calc_action.triggered.connect(self.start_inf_calc)
        toolbar.addAction(start_inf_calc_action)

        stop_inf_calc_action = QAction("Остановить бесконечный расчет", self)
        stop_inf_calc_action.triggered.connect(self.stop_inf_calc)
        toolbar.addAction(stop_inf_calc_action)

        self.time_edge = QDoubleSpinBox(maximum=1000)
        
        self.time_edge.setDecimals(2)
        self.time_edge.setValue(1.0)
        


        self.canvas = MplCanvas()

        toolbar = NavigationToolbar2QT(self.canvas, self)

        self.timer = QTimer()
        self.timer.timeout.connect(self.inf_calc)
        self.flag = True
        self.calculated = False
        layout = QVBoxLayout()
        layout.setSpacing(10)


        layout.addWidget(self.label)
        layout.addWidget(self.time_edge)
        layout.addWidget(toolbar) 
        layout.addWidget(self.canvas)

        layout.addStretch()   # прижимает все к верху

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.count = 0


    def calculate(self):
        if self.json_loaded and self.obj_loaded:
            self.solver.T_0 = self.T_0
            self.solver.t_span = (0, self.time_edge.value())
            self.solver.solve_ode()
            
            self.canvas.ax.clear()
            for i in range(len(self.solver.sol.y)):
                self.canvas.ax.plot(self.solver.sol.t, self.solver.sol.y[i], label="T" + str(i + 1))

            self.canvas.ax.legend()
            self.canvas.draw()
            self.calculated = True
            
        
    def start_inf_calc(self):
        if self.calculated:
            self.canvas.ax.clear()
            self.t = 0
            self.solver.T_0 = self.T_0
            self.calculated = False
            self.flag = True
        if self.json_loaded and self.obj_loaded:
            self.timer.start(100)

    def stop_inf_calc(self):
        if self.json_loaded and self.obj_loaded:
            self.timer.stop()

    def inf_calc(self):
        if self.json_loaded and self.obj_loaded:
            colors = ['g', 'r', 'k', 'b', 'y']
            self.solver.t_span = (self.t, self.t + self.dt)
            self.solver.solve_ode()
    #        self.solver.T_0 = self.solver.sol.y[:, self.solver.sol.y.shape[1] // 2]
    #        self.t += self.dt / 2
            self.count += 1

            
            

            self.solver.T_0 = self.solver.sol.y[:, -1]
            self.t += self.dt
            for i in range(len(self.solver.sol.y)):
                self.canvas.ax.plot(self.solver.sol.t, self.solver.sol.y[i], label="T" + str(i + 1), color = colors[i])
                if self.count > 5:
                    self.canvas.ax.set_xlim(self.count * self.dt - 4, self.count * self.dt)
            if self.flag:
                self.canvas.ax.legend()
                self.flag = False
            
            self.canvas.draw()


    def open_obj(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать OBJ файл",
            "",
            "OBJ Files (*.obj)"
        )
        if file_path:
            self.obj_loaded = True
            self.solver.load_obj(file_path)

    def open_json(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать json файл",
            "",
            "json Files (*.json)"
        )
        if file_path:
            self.json_loaded = True
            self.solver.load_json(file_path)
            self.T_0 = self.solver.T_0
    

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()