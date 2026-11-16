from dataclasses import dataclass

class Animal:
    def __init__(self, name, legs):
        self.name = name
        self.legs = legs
    def describe(self):
        return f"{self.name} has {self.legs} legs"
    def grow(self):
        self.legs += 1

class Dog(Animal):
    def __init__(self, name):
        self.name = name
        self.legs = 4
    def speak(self):
        return "woof"

@dataclass
class Point:
    x: int
    y: int
    def dist2(self):
        return self.x * self.x + self.y * self.y

a = Animal("spider", 8)
a.grow()
print(a.describe())
d = Dog("rex")
print(d.describe(), d.speak())
p = Point(3, 4)
print(p.dist2(), p.x, p)
q = Point(3, 4)
print(p == q)
pts = [Point(i, i * 2) for i in range(3)]
pts[0].x = 99
print([pt.x for pt in pts])
alias = pts[1]
alias.y = -1
print(pts[1].y)
