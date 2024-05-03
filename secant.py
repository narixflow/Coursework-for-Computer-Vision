import math


def secant(func, x0, x1):
    print(x0, x1)

    M_steps = 30
    delta = math.pow(10.0, -8)
    epsiron = math.pow(10.0, -8)

    for iter in range(M_steps):
        print("iter=", iter, x0, x1, abs(x1 - x0) < epsiron, epsiron)
        if abs(x1 - x0) < epsiron:
            break

        if abs(func(x1) - 0) < delta:
            break

        fen_mu = (func(x1) - func(x0)) / (x1 - x0)
        x_new = x1 - func(x1) / fen_mu

        x0, x1 = x1, x_new


    return x1


def f_x_q3(x):
    return -(math.pow(x, 3)  + math.cos(x))

if __name__ == "__main__":

    root = secant(f_x_q3, -1.0, 0.0)
    print("root = ", root)



