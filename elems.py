import numpy as np

class point:
    x : np.float64
    y : np.float64
    z : np.float64

    def __init__(self, x : np.float64, y : np.float64, z : np.float64):
        self.x = x
        self.y = y
        self.z = z

class face:
    x : np.int32
    y : np.int32
    z : np.int32

    def __init__(self, x : np.int32, y : np.int32, z : np.int32):
        self.x = x
        self.y = y
        self.z = z


class element:
    points : list[point]
    faces : list[face]
    surface_area : np.float64

    def __init__(self, points : list[point], faces : list[face]):
        self.points = points
        self.faces = faces
        self.calc_surface_area()

    def calc_triangle_area(self, fc : face):
        p1 = fc.x
        p2 = fc.y
        p3 = fc.z

        a = np.array([self.points[p2].x - self.points[p1].x, self.points[p2].y - self.points[p1].y, self.points[p2].z - self.points[p1].z])
        b = np.array([self.points[p3].x - self.points[p1].x, self.points[p3].y - self.points[p1].y, self.points[p3].z - self.points[p1].z])
        return (1 / 2) * np.linalg.norm(np.cross(a, b))

    def calc_surface_area(self):
        self.surface_area = np.array([self.calc_triangle_area(face) for face in self.faces]).sum()
        return self.surface_area

class device:
    elements : list[element]
    el_surf_cross : np.array
    el_surf : np.array
    def __init__(self, filename : str):
        self.elements = []
        flag = False
        cnt = 0

        with open(filename, 'r') as f:
            points : list[point] = []
            faces : list[face] = []
            for line in f:
                if line[0] == 'v' and flag:
                    self.elements.append(element(points, faces))
                    cnt += len(points)
                    points : list[point] = []
                    faces : list[face] = []
                    flag = False
                if line[0] == 'v':
                    st = np.float64(line.strip('v ').split())
                    points.append(point(st[0], st[1], st[2]))
                elif line[0] == 'f':
                    st = np.int32(line.strip('f ').split()) - cnt - 1
                    faces.append(point(st[0], st[1], st[2]))
                    flag = True
            if flag:
                self.elements.append(element(points, faces))
        
        self.el_surf_cross = np.zeros((len(self.elements), len(self.elements)))
        self.find_elements_cross_sectional_areas()
        self.el_surf = np.array([el.surface_area for el in self.elements])

    def intersect_elements(self, el1 : element, el2 : element):
        sm = 0
        for face1 in el1.faces:
            for face2 in el2.faces:
                p11 = el1.points[face1.x]
                p12 = el1.points[face1.y]
                p13 = el1.points[face1.z]
                p21 = el2.points[face2.x]
                p22 = el2.points[face2.y]
                p23 = el2.points[face2.z]

                v1 = np.sort(np.array([p11.x, p11.y, p11.z, p12.x, p12.y, p12.z, p13.x, p13.y, p13.z]))

                v2 = np.sort(np.array([p21.x, p21.y, p21.z, p22.x, p22.y, p22.z, p23.x, p23.y, p23.z]))
                
                if np.abs(v2 - v1).max() < 1e-2:
                    a = np.array([p12.x - p11.x, p12.y - p11.y, p12.z - p11.z])
                    b = np.array([p13.x - p11.x, p13.y - p11.y, p13.z - p11.z])
                    sm += (1 / 2) * np.linalg.norm(np.cross(a, b))

        return sm

    def find_elements_cross_sectional_areas(self):
        for i in range(len(self.elements)):
            for j in range(i + 1, len(self.elements)):
                el1 = self.elements[i]
                el2 = self.elements[j]
                self.el_surf_cross[i][j] = self.intersect_elements(el1, el2)
                self.el_surf_cross[j][i] = self.el_surf_cross[i][j]


                    



                    
                    

