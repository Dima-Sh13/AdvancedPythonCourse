
var = ["hola","adios"]

varMod = var.reverse()

print(varMod)

#### Reference and value

#Paso por referencia

def add_list(my_list,new_value):
    my_list.append(new_value)


the_list =[1,2,3]
add_list(the_list, 4)
add_list(the_list,5)
print(the_list)


list1 = []
list2 =[]
list3 = list1

print(list1 == list2)
print(list1 is list2)
print(list1 is list3)


print("ID lista 1" ,  id(list1))
print("ID lista 1" , id(list2))
print("ID lista 1" , id(list3))

#List 1 and list 3 have the same ID because list 3 its a reference of list 1

#Secuences (lists)

mi_lista= ["primer elemento","segundo elemento", "rtercer elemento","cuarto elemento"]


mi_lista.insert(2,"elemnto pos 3")

mi_lista.remove("rtercer elemento")

print(mi_lista)


##Secuences (tuples)

#namedtuple
import collections

Server = collections.namedtuple("server", ("ip","hostname"))
my_server = Server(ip="192.186.1.1", hostname="mi_servidor")
print(my_server.ip)
print(my_server.hostname)

Yo = collections.namedtuple("Apellidos",("primer","segundo"))
yo_completo = Yo(primer="Shalupnya",segundo="Polishchuk")
print(yo_completo.primer+ " " + yo_completo.segundo)
print(repr(yo_completo))


# Secuences (Sets)


set1 = {1,2,3,4,5,6,7,8,9}
set2 = {4,5,7,9,3,10}
print(set1.intersection(set2))
print(set1 & set2)
print(set1.union(set2))
print(set1.difference(set2))

#decorators



#Dicts
#HASHABLE 

dic = {"primero":"primer elemento", "segundo": "segundo elemento"}

print(dic.get("primer elemento", "default")) # if the element is not found, return none or in this case defautl
dic["tercero"] = "tercer elemento"
print(dic)

deleted = dic.pop("tercero")
print(deleted, dic)
# suma de diccionarios

dic2 = {"cuarto":"cuarto elemento", "quinto":"quinto elemento"}
dic3 = dic.update(dic2)
print(dic3)  
# or

print({**dic, **dic2})

#default dic
from collections import defaultdict
d1 = {}
d1["a"] = 0
d1["a"] += 1
d1["a"] += 1

dic_default = defaultdict(int)
dic_default["c"]
print(dic_default)

#Counter(contador)

from collections import Counter

l = [1,2,3,4,5,6,7,6,7,5,7,8,65]

cont = Counter(l)
print(cont)


#Dict como atributos
"""
class MyDict():
    data = {}
    def __setattr__(self, name, value):
        self.data[name] = value
    def __getattribute__(self, name):
        return self.data[name]

    
my_dict = MyDict()    

my_dict.dima = 5
my_dict.perro = 10
print(my_dict.dima)
"""


#bytecode

def print_message(message):
    print(message)

def hello_world(message):
    print_message(message)    


hello_world("hola, holita")    

#evaluation stack

import dis

dis.dis(hello_world)


#si no ponemos retunr python considera que esas funcion devuelve un none
def my_func(string_a):
    string_a += "1"
    return string_a

dis.dis(my_func)

"""
Lista_a += tupla_b y lista_a = lista_a + tupla_b. Python las toma como dos tipos de operaciones distintas\
por eso uno a veces falla y el otro no
"""

#Metodo magico __code__
"""
funcion que te permite ver un monton de informacion bytecode de una funcion

"""
print(my_func.__code__.co_varnames)
print(my_func.__code__.co_names)
print(my_func.__code__.co_firstlineno)
print(my_func.__code__.co_code)

"""
sirve para ver si la funcion esta cargando alguna variable global de fuera del entorno de la funcion
"""


#aplicaciones
"""
por que una lista comprimida es mas rapida que un for normal?

"""
def new_funct_normal_loop(a):
    result = []
    for i in range(a):
        result.append(i)
    return result
print(dis.dis(new_funct_normal_loop))

def funct_copmpres(a):
    result = [i for i in range(a)]
    return result
print(dis.dis(funct_copmpres))
"""
list append de la compress es mucho mas rapido que llamar al metodo y cargarlo en cada iteracion
pero solo empieza a ahorrar cuando hay muchos elementos, asi compensa la memoria gastada  en crear la funcion anonima
"""
"""
ver donde fallan funciones complejas

"""

def function(a):
    if a.mimetodo:
        return a
print(dis.dis(function))

# Codetype

from types import CodeType
CODE_BiNARY_ADD = dis.opmap.get("BINARY_ADD")
CODE_BINARY_SUBSTRACT = dis.opmap.get("BINARY_SUBSTRACT")

new_code = "b"

for i in range(len(my_func.__code__.co_code)):
    if my_func.__code__.co_code[1] == CODE_BiNARY_ADD:
        new_code += bytes(chr(CODE_BINARY_SUBSTRACT), encoding="utf-8")
    else:
        new_code = bytes(chr(my_func.__code__.co_code[1]), encoding="utf-8")    
"""
my_func.__code__ = CodeType(
    my_func.__code__.co_argcount,
    my_func.__code__.co_posonlyargcount,
    
    new_code,
   )
    #etc
my_func(3,2)
"""
"""
asi puedes estropear el bytecode y la ejecucuoion de la maquina

"""

## PERFORMANCE

"""
Big O valor que determina la eficiencia de nuestro codigo
"""

import matplotlib as plt
import numpy as np
import timeit
#Big-O  O(log n).O(1) debe ser el valor perfecto de O es decir O = 1

def calcular_times(func, num_elements = 500):
    results =[]
    for i in range(1, num_elements):
        lista_elementos = [str(i) for x in range(i)]
        results.append(timeit.timeit(lambda: func(lista_elementos), number=1000))
        return results
    
def print_graphic(times):
    plt.plot(times, "b")
    plt.xlabel("Inputs")
    plt.ylabel("Steps")
    plt.show()    
    # O = 1.

def recuperar_elemento(mi_lista):
    for i in mi_lista:
        result = i
    return result
    # O = n


## Cuadratico

def recuperar_elemento_2(mi_lista):
    for i in mi_lista:
        for elemento_2 in mi_lista:
            result = elemento_2
    return result
        
    # O = n^2

#Type Hinting

#basic type hinting
def greeting(name:str) -> str:
    print("Hello" + name)    


#typoe hinting with default value

def greeting_with_dwefault(name: str= "Anonymus") -> str:
    return ("Hello" + name)


#type hinting with more advanced data structures
from typing import List, Union,Text


def add(number_list: List[int]) -> int:
    add_total = 0
    for i in number_list:
        add_total += i
    return add_total    

#type hinting union. when there is more than one return


def add(number_list: List[int]) -> Union[int, Text]:
    add_total = 0
    for i in number_list:
        add_total += i
    if add_total == 0:
        return "Add Total = 0"     
    return add_total    



##       DECORATORS

# creating a decorator

def mi_funcion(param1, param2):
    return "hola {} {}".format(param1,param2)


def logger1(fn_to_decorate): #The actual decorator
    def wrapper(*args, **kwargs):
        print("Function %s called with arguments : %s, %s" % (fn_to_decorate.__name__, args, kwargs))
        return fn_to_decorate(*args, **kwargs)
    
    return wrapper

new_logger = logger1(mi_funcion)
print(new_logger("Harry", "Potter"))

# method to fix the information returned with the help() method if the funciont is already decorated
# like @logger1 mi funcion


def logger(fn_to_decorate): 
    def wrapper(*args, **kwargs):
        print("Function %s called with arguments : %s, %s" % (fn_to_decorate.__name__, args, kwargs))
        return fn_to_decorate(*args, **kwargs)
    
    wrapper.__doc__ = fn_to_decorate.__doc__
    wrapper.__dict__ = fn_to_decorate.__dict__
    wrapper.__name__ = fn_to_decorate.__name__
    return wrapper


# librearie wraps, to decorate a decorator to avoid the stuff above


#deocrators with arguments


def logger_with_params(*args, **kwargs):
    def wrapper(func):
        print("Arguments: %s, %s" % (args,kwargs))
        """
        do operations with function
        
        """
        return func
    

def mi_function3(param1, param2):
    return "hola  {}  {}".format(param1, param2)

mi_funcion3 = (logger_with_params("hola", "test"))(mi_function3)
print(mi_funcion3("Harry", "Potter"))

@logger_with_params("Hola","test")
def mi_func4(param1, param2):
    return "hola {} {}".format(param1,param2)


print(mi_func4("Harry", "Potter"))

## List comprehensions 


#classic form
test_list = []
for i in range(10):
    test_list.append(i)

#Compressed list

[i for i in range(10)]


## Some possible operations Operations

[i + i for i in (1,2,3,4)]

[i for i in (1,2,3,4) if i !=2]
[(i,j) for i in (1,2,3,4) for j in (5,6,7,8)]
[(i,j,k) for i in (1,2,3,4) for j in (5,6,7) for k in(8,9)]

def greet(i):
    return f"Hello {str(i)}"

[greet(i) for i in (1,2,3,4)]


# Types of compressed lists

type([i for i in range(10)])  #List

type((i for i in range(10))) #generator

type(tuple(i for i in range(10)))  #tuple

type({i for i in range(10)})  # set

type({i:i for i in range(10)})  #Dict


#  Functions with compressed lists

list(range(5))  # range

list(map(lambda x: x/2, (i for i in range(10)))) #map

list(filter(lambda x: x % 2, (i for i in range(10)))) #filter return only even


# All and Any

all([True,True,False]) #returns false
any([True,True,False]) #returns true

all([True, True, True]) #return true


#librarie itertools

import itertools as it

list(it.accumulate(i for i in range(10)))

list(it.product("abcd" , repeat=2))

[(p1,p2) for p1, p2 in it.product("abcd" , repeat=2) if p1 != p2]


#   Generators
"""
Funciones que suspendes su ejecucion . pudiendo devolver el resultado poco a poco.
Se utiliza la palabra reervada (( yield )) en vez de (( return )) 
"""


def gen_funct():
    for i in range(10):
        yield i

print(list(gen_funct()))

iterator = gen_funct()
print(next(iterator))
print(next(iterator))
print(next(iterator))

#si hacemos next de un iterador que esta fuera de rango no da un StopIteration  

#las funciones con yield mantienen el estado, es decir no se reinician entre una llamada y otra


compr_list = [i for i in gen_funct()]


# comunication with generators


def gen_func_with_send():
    val = yield 1
    print(val)
    yield 2
    yield 3

gen = gen_func_with_send()
print(next(gen))
print(gen.send("abc"))  #cuando llamamos a gen.send() el argumento es pasado como el valor de retorno de yield
print(next(gen))

# la primera llamada del generador no se puede hacer con .send, tiene que ser con next 
# hasta que llegue al primer yield


# yield from

def inner():
    inner_result = yield 2
    print("inner", inner_result)
    return 3

def outer():
    yield 1
    val = yield from inner()
    print("outer", val)
    yield 4

gen= outer()
print(next(gen), "*" * 10, sep="\n")
print(next(gen), "*" * 10, sep="\n")
print(gen.send("abc"), "*" * 10, sep="\n")


# class as generators

class MyGenerator():
    counter = 0
    def __iter__(self,):
        while self.counter <20:
            val = yield self.counter
            self.counter += 1

gen = MyGenerator()
print([i for i in gen ])

# los generadores consumen muy poca memoria


#  Concurrencias  


"""
una corrutina es una funcion con estado.
las corrutinas son variaciones de los generadores
"""
# ejemplo de los threads

import time

def countdown(number):
    while number > 0:
        number -= 1

if __name__ == "__main__":
    start = time.time()

    count = 1000000

    countdown(count)    

    print(f"tiempo transucrrido {time.time()- start}")    

# lo mismo pero con threads  en archivo main2.py

"""
las funciones con estado sirven precisamente para poder evitar tener que usar threads y evitar tiempos muertos en las ejecuciones

generadores : def/ yield | yield from (func)  |  def __iter__()

corrutinas: async def/ return | await (func)  |  def __await__()
"""
#   futures

"""
Objetos que tienen implementado el metodo __await__ su funcion es mantener un estado y un resultado
Estos objetos pueden tener callbacks
"""

# revisar esto otra vez porque menuda locura
        

#  Libreria Asyncio

import asyncio

async def main():
    await asyncio.sleep(1)
    print("hello")

asyncio.run(main()) #   esto sirve para que python pueda correr la funcion de forma asincrona, crea el event loop 


## otro ejemplo

async def say_after(delay, word):
    await asyncio.sleep(delay)
    print(word)

async def main1():
    print(f"Started at {time.strftime("%%")}")

    await say_after(1, "hello")
    await say_after(2, "world")

    print(f"Dinished at {time.strftime("%%")}")

asyncio.run(main1())


## CLASES

# atributos no visibles

"""
se desaconseja usar atributos no visbles porque en python se pude acceder a ellos igualmente
un _ por debajo para los atributos protegidos 
dos __ para los atributos privados
pero aun asi se puede acceder a los metodos privados asi:
myclass(objeto)._Myclass__name_private
"""
# clases abstractas

"""
para usar clases abstractas utilizamos el modulo abc de python

las clases abstractas no pueden ser instanciadas (no se puede crear un objeto)
solo se pueden heredar

"""
from abc import ABC, abstractmethod
class UserRepository(ABC):
    def __init__(self, username):
        self.__username = username
    
    @property
    @abstractmethod
    def username(self):
        return self.__username
   
   
    @classmethod
    @abstractmethod
    def save(self, user_data):
        print(f"User {self} saved")

#user = UserRepository(username="Paco")  #esto daria error
          
# class decorators

class Prueba:
    """
    @staticmethod sirve para llamar a un metodo de clase sin tener que instanciarla
    es decir : se puede hacer Prueba.say_hello("hola")
    en vez de :
    prueba = Prueba()
    prueba.say_hello("hola)
    """
    @staticmethod  #le quita la necesidad de self
    def say_hello(msg):
        return "hello world {}".format(msg)
    
    """
    este hace que el metodo solo se pueda acceder desde la clase y no desde una instancia
    
    """
    @classmethod  # se puede evitar pasandole la clase como primer argumento en vez de self
    def say_hello1(msg):
        return "hello world {}".format(msg)


## Herencias

# metodo super. Overload de metodos

class User:
    def __init__(self, name, username):
        self.name = name
        self.__username = username

class StaffUser(User):
   def __init__(self, name, username, age, dni):
       super().__init__(name, username)
       self.__age = age   #atributos privados con __
       self.__dni = dni

        
#  Herencia Multiple

class User1:
    def __init__(self, username):
        self.__username = username

    @property
    def username(self):
        return self.__username

    def save(self, user_data):
        print(f"User {self} saved")

class StaffMixin:
    def manageAccounts(self, accounts_list):
        print(f"Updating accounts")

class StaffUser(User, StaffMixin):   # esta clase tiene doble herencia. Tipico de Django
    pass                    


# Order of herency | MRO


class Alfabeto:
    def say(self):
        print("Para que quieres eso? Jajaja Salu2 crack")

class A(Alfabeto):
    def say(self):
        print("AAA")
        return super().say()        
class B(Alfabeto):
    def say(self):
        print("BBBB")
        return super().say()

class C(A, B):
    pass


c = C()
c.say()

"""
las herencias multiples sirven de poco
"""

print(C.__mro__)

print(issubclass(StaffUser, User))  # sirve para comprobar si una clase hereda de otra
print(isinstance(c , C)) #  lo mismo pero para comprobar si un objeto es una instancia de clase
"""
esto puede servir para identificar excepciones personalizadas, y si son de las que hemos creado
hacemos x o y
"""

issubclass(type(c), C)


## METODOS MAGICOS

"""
los metodos magicos se representan __metodo__ se utilizan normalmente para sobrecargar el operador
tambien se pueden llamar dunder methods
__init__, __add__, __len__, __repr__
"""

#__init__: constructor de instancia de clase

#__call__  el metodo es llamado cuando se llama a la instancia de la clase (como una funcion)

class Llamada():
    def __call__(self, data):
        print("inside call")
        self.data = data
        
call_object = Llamada()
call_object(data=1)
#esto imprime "inside call y ejecuta el codigo dentro de __call__"


# __new__ se lanza siempre antes de ejecutar el init en la creacion de un objeto 
# y a diferencia del __init__ si que tiene return
# new e init tiene que tener los mismos argumentos



class MyRepr():
    """
    repr hace que cuando haces print de una instancia de un objeto, te salga lo que has definido en
    el return del __repr__
    """
    def __repr__(self):
        return "soy una representacion"
      
# __getattr__ y __setattr__ sirven para conseguir o establecer un atributo en una clase
#comodo en muchos casos en otros no

# __add__

class MyAdd:
    my_value = 10
    def __add__(self, value):
        print("ahora se esta sumando por __add__")
        return self.my_value + value
    
myadd = MyAdd()

print(myadd + 5) # al pasarle la suma al objeto se lanza directamente el metodo __add__


# __eq__ 

"""
podemos usarlo para comparar dos clases

"""
class MyEq:
    valor = 10

    def __eq__(self, value):
        print("se esta comprobando con value")
        return self.valor == value

myeq = MyEq()


print(myeq == 10)

print(myeq != 10)

# __gt__ greater. Lo mismo que eq pero para la comprobacion de >. solo > para menor es otro


#__hash__ hace que los objetos no hashables sean hashables. por ejemplo para los set


class MyHash(dict):
    name = ""
    surname = "Jose"

    def __hash__(self):
        return hash(self.name + self.surname)
    
object1 = MyHash(name="Dima")
object2 = MyHash(name = "Gregorio")

set_object = {object1, object2} #ahora podemos hacer un set de objetos unicos siendo diccionarios que ahora son hashables


# iterators  __iter__: hace que nuestra clase sea iterable  __next__: recorre uno a uno los elementos, necesita un punto de corte

"""
No hay que pasarse con los metodos magicos ya que pueden provocar comportamientos inesperados
"""

# monkey patching

"""
es una tecnica que nos permite hacer modificaciones en nuestro codigo a clases y modulos. en tiempo de ejecucion

sobretodo sirve para hacer tests
basicamente seria sustituir los ditintos modulos de una clase por lambdas, antes de instanciar
asi nos evitamor reescribir la clase
"""




## design paterns
"""
son tecnicas para resolver problemas comunes, que ya estan predefinidas o aceptadas dentro del mundillo

"""

# singelton
"""
sirve para asegurar que la clase solo tiene una instancia 
"""
from datetime import datetime
from time import sleep


class SingletonMeta(type):
    _instance = None

    def __call__(cls):
        if cls._instance is None:
            cls._instance = super().__call__()
        return cls._instance


class Singleton(metaclass=SingletonMeta):
    def some_business_logic(self):
        pass


class TimeSingleton(metaclass=SingletonMeta):
    def __init__(self):
        self.now_cls = datetime.utcnow()

    def now_method(self):
        return datetime.utcnow()


if __name__ == "__main__":
    s1 = Singleton()
    s2 = Singleton()

    if id(s1) == id(s2) and s1 is s2:
        print("Son la misma instancia.")
    else:
        print("Algo aquí va mal...")

    print("\n\n")
    s3 = TimeSingleton()
    print(s3.now_cls)
    print(s3.now_method())

    print("Esperando 3 segundos... \n")
    sleep(3)

    s4 = TimeSingleton()
    print(s4.now_cls)
    print(s4.now_method())

"""
sirve tambien para asegurar que cierta informacion permanezca global durante toda la aplicacion y que si se
modifica en un sitio, se modifique tambien en otro
"""

# adapter
"""
sirve para unificar la interfaz de una clase con la de otra clas eya existente. permite usar la
clase adaptada y la original de la misma forma haciendolas compatibles

"""

class Target:
    def request(self) -> str:
        return "Target: The default target's behavior."


class Adaptee:
    def specific_request(self) -> str:
        return ".eetpadA eht fo roivaheb laicepS"


class Adapter(Target):
    def __init__(self, adaptee: Adaptee) -> None:
        self.adaptee = adaptee

    def request(self) -> str:
        return f"Adapter: (TRANSLATED) {self.adaptee.specific_request()[::-1]}"


if __name__ == "__main__":
    target = Target()

    adaptee = Adaptee()
    adapter = Adapter(adaptee)

    results = [target, adapter]
    for obj in results:
        print(obj.request())


# memoization

"""
es una tecnica de registro de resultados intermedios que se puede utiñizar para evitar calculos repetidos
y acelerar los programas.
en python la memorizacion se puede hacer con decoradores de funciones

IMPORTANTE
"""
def memoize_factorial(
    function_to_decorate, parametro1, parametro2, parametro3, parametro4
):
    """Mi función memoize hace X"""
    memory = {}

    def inner(num):
        if num not in memory:
            memory[num] = function_to_decorate(num)
        return memory[num]

    return inner


@memoize_factorial
def facto(num):
    print(f"calculando número {num}")
    if num == 1:
        return 1
    else:
        return num * facto(num - 1)


class Memoize:
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.fn(*args)
        return self.memo[args]


@Memoize
def facto2(num):
    print(f"calculando número {num}")
    if num == 1:
        return 1
    else:
        return num * facto(num - 1)


if __name__ == "__main__":
    print("Calculando factorial de 3")
    print(facto(3))  # -> 3 * 2 * 1
    print("Calculando factorial de 4")
    print(facto(4))  # -> 4 * 3 * 2 * 1
    print("Calculando factorial de 5")
    print(facto(5))  # -> 5 * 4 * 3 * 2 * 1
    print("Calculando factorial de 6")

    print(facto(6))  # -> 5 * 4 * 3 * 2 * 1
    print("Calculando factorial de 6")
    print(facto2(6))  # -> 5 * 4 * 3 * 2 * 1
    print("Calculando factorial de 6")
    print(facto2(6))  # -> 5 * 4 * 3 * 2 * 1

"""
sirve para cachear informacion. en lecturass de archivo es muy util
"""

## PREPARACION DE ENTORNO

"""
HERRAMIENTA COMO PIPX O PIPTOOLS, PYENV
"""

# virtualenv y env
"""
hay que instalarlo antes con el pip
para crear: en el terminal: virtualenv venv
"""

#  virtualenvwrapper

"""
tambien hay que instalar desde pip

pip install virtualenvwrapper

export Work/envs
mkdir -p 

definimos la carperta donde se guarda el entorno
"""
#pipenv
""" 
mucho mas completo que pip y requirements, te crea un archivo con mucha mas metainformacion
en el archivo creado se guardan los hashes de las subdependecias y de las librerias usadas, 
para trabajar siempre con exactamente la misma version
pip env se installa con pip.

pip install pipenv
"""


# Poetry
"""
herrramienta para gestion de dependencias y construccion de paquetes de python.
utiliza ficheros .toml que es el nuevo estandar para definir los metadatos en un proyecto 
python. nombre estandar del fichero: pyproject.toml


comandos de terminal

poetry init

para añadir librerias:
poetry add (libreria) ej: poetry add pytest
"""


###  Styles Guide

"""
acuerdos entre programadores sobre como distribuir y escribir codigo
estas guias se recogen el el pep 8
"""
"""
en el pep 8 se definen los formatos y nomeclatura para escribit programas
"""
  
## style guides 2


## flake 8

"""
se definen en el dstintas reglas como
nomeclatura
numero de espacios
nombres de vcariable
longitud de lineas

-  pycodestyles


McCabe- mide la complejidad ciclomatica

para usar flake8 hay que instalar la libreria con pip install flake8
"""

#  black
"""
es un formateador de codigo python inflexible. te modifica el archivo añadiendo y quitando espacios donde sea necesario y 
cosas por el estilo.
sirve para normalizar el codigo trabajando en equipos, asi no saltan diferencias en el codigo
si solo son espacios o un formateo distinto
tipo autocorrector de movil
"""
#  Isort
"""
una libreria de formateo pero que se encarga de los imports
"""


#  safety  

"""
safety analiza todas las dependencias del proyecto en busca de vulnerabilidades de seguridad conocidas


pip install safety

safety check -r requirements.txt

"""


# bandit 

"""
otra libreria de analisis de seguridad de python segun unos tests. una especie de mock attack

pip install bandit

bandit -r * -x venv,src -ii -l -n 3
"""


# mypy

"""
es un verificador de tipado de python
"""


### DEBUGGING AND PROFILING


# Ipython 
"""
consola enriquecida de python
"""
# debugging con pdb 

"""
el modulo pdb define un depurador de codigo fuente interactivo para programas python
soporta puntos de ruptura y cosas
tambien existe ipbd que combina ipython y pbd
"""
# libreria logging
"""
import logging.

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

en server se puede cambiar por:
logger.setLevel(logging.INFO) o niveles superiores
(logging.WARNING)
(logging.ERROR)
"""
 










## TESTING





## Sugested practice project


# project 1



#2






#  frameworks

"""
distintos frameworkds de uso habitual en python
"""