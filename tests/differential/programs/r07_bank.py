class Account:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
        self.history = []
    def deposit(self, amt):
        if amt <= 0:
            return False
        self.balance += amt
        self.history.append(("dep", amt))
        return True
    def withdraw(self, amt):
        if amt > self.balance:
            self.history.append(("fail", amt))
            return False
        self.balance -= amt
        self.history.append(("wd", amt))
        return True
accts = {}
for name in ["ann", "bob"]:
    accts[name] = Account(name)
accts["ann"].deposit(100)
accts["ann"].withdraw(30)
accts["bob"].withdraw(5)
accts["bob"].deposit(-1)
for name in accts:
    a = accts[name]
    print(a.owner, a.balance, a.history)
def transfer(src, dst, amt):
    if src.withdraw(amt):
        dst.deposit(amt)
transfer(accts["ann"], accts["bob"], 50)
print(accts["ann"].balance, accts["bob"].balance)
