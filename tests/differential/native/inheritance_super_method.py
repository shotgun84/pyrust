class A:
    def f(self):
        return 1
class B(A):
    def g(self):
        return self.f() + 1
print(B().g())
