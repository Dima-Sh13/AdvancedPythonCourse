import time
import threading

def countdown(number):
    while number > 0:
        number -= 1

if __name__ == "__main__":
    start = time.time()

    count = 1000000        

    t1 = threading.Thread(target=countdown, args=(count,))
    t2 = threading.Thread(target=countdown, args=(count,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print(f"tiempo transcurrido {time.time()- start}")


"""
El gil de python evita que se ejecuten los dos thread en paralelo

para poder ejecutar en paralelo tenemos que usar la libreria multiprocesing

se puede trabajar con threads en python pero tienenq que suspender la ejecucion de otro threads
"""    
import multiprocessing

t1 = multiprocessing