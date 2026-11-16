class A:
    def f(self):
        return "A"
class B(A):
    def f(self):
        return "B"
xs = [A(), B()]
for x in xs:
    print(x.f())
