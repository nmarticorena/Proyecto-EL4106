# Proyecto-EL4106
Proyecto-EL4106 Este consiste en la implementacion de algoritmos geneticos para el calculo de la cinematica inversa de un brazo robotico de 4dof.

# Integrantes
Javier Urrutia R.
Nicolás Marticorena V.

# Links
[Repo brazo 4dof](https://github.com/JavierUR/SimpleArm)

[Meeting 1](https://github.com/JavierUR/Proyecto-EL4106/blob/master/doc/meeting_1.md)


# Instrucciones

En primer lugar se debe acceder al directorio src/fitness y compilar el codigo c++ utilizado para el calculo de las cinematica directa.

```
cd src/fitness/
make
```

Una vez que se compile se puede ejecutar el codigo del algoritmo CGA mediante el siguente comando

```
python CGA_Trayectoria.py <x,y,z> <o1,o2,o3,o4>
```
Con xyz los puntos cartesianos deseados finales y o1,o2,o3 y o4 los angulos iniciales del brazo.

un ejemplo de como escribirlo se encuentra a continuación:

```
python CGA_Trayectoria.py 1,1,1 0,0,0,0
```
