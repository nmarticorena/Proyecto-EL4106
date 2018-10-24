# Proyecto-EL4106
Proyecto-EL4106 Este consiste en la implementacion de algoritmos geneticos para el calculo de la cinematica inversa de un brazo robotico de 4dof.

# Links
[Repo brazo 4dof](https://github.com/JavierUR/SimpleArm)


# Meeting 1

## Algoritmo Evolutivo
### Definiciones
#### Codificación
Pendiente
#### Operador Genetico de selección

#### Croosover
#### Mutación
#### Fitness

## Cinematica Directa
Para la cinematica directa se utilzo la formulacion 'Denavit-hartenberg' En la cual tenemos los valores parametrizados de:

Largo=0.5+0.1 mt

altura=0.1 mt

### Traslación de ejes
<a href="https://www.codecogs.com/eqnedit.php?latex=T_t=\begin{bmatrix}&space;0&space;&&space;0&space;&&space;0&space;&&space;0\\&space;0&space;&&space;0&space;&&space;0&space;&&space;0\\&space;0&space;&&space;0&space;&&space;0&space;&&space;largo\\&space;0&space;&&space;0&space;&&space;0&space;&&space;1&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_{traslacion}=\begin{bmatrix}&space;0&space;&&space;0&space;&&space;0&space;&&space;0\\&space;0&space;&&space;0&space;&&space;0&space;&&space;0\\&space;0&space;&&space;0&space;&&space;0&space;&&space;largo\\&space;0&space;&&space;0&space;&&space;0&space;&&space;1&space;\end{bmatrix}" title="T_t=\begin{bmatrix} 0 & 0 & 0 & 0\\ 0 & 0 & 0 & 0\\ 0 & 0 & 0 & largo\\ 0 & 0 & 0 & 1 \end{bmatrix}" /></a>


### Traslación de base
<a href="https://www.codecogs.com/eqnedit.php?latex=T_t=\begin{bmatrix}&space;0&space;&&space;0&space;&&space;0&space;&&space;0\\&space;0&space;&&space;0&space;&&space;0&space;&&space;0\\&space;0&space;&&space;0&space;&&space;0&space;&&space;altura\_base\\&space;0&space;&&space;0&space;&&space;0&space;&&space;1&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_{traslacion}=\begin{bmatrix}&space;0&space;&&space;0&space;&&space;0&space;&&space;0\\&space;0&space;&&space;0&space;&&space;0&space;&&space;0\\&space;0&space;&&space;0&space;&&space;0&space;&&space;altura\_base\\&space;0&space;&&space;0&space;&&space;0&space;&&space;1&space;\end{bmatrix}" title="T_t=\begin{bmatrix} 0 & 0 & 0 & 0\\ 0 & 0 & 0 & 0\\ 0 & 0 & 0 & altura\_base\\ 0 & 0 & 0 & 1 \end{bmatrix}" /></a>

### Rotacion en eje y (brazos)
<a href="https://www.codecogs.com/eqnedit.php?latex=Rot_y=\begin{bmatrix}&space;Cos(\theta)&space;&&space;0&space;&&space;Sen(\theta)&space;&&space;0\\&space;0&space;&&space;1&space;&&space;0&space;&&space;0\\&space;-Sen(\theta)&space;&&space;0&space;&&space;Cos(\theta)&&space;0\\&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Rot_y=\begin{bmatrix}&space;Cos(\theta)&space;&&space;0&space;&&space;Sen(\theta)&space;&&space;0\\&space;0&space;&&space;1&space;&&space;0&space;&&space;0\\&space;-Sen(\theta)&space;&&space;0&space;&&space;Cos(\theta)&&space;0\\&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;\end{bmatrix}" title="Rot_y=\begin{bmatrix} Cos(\theta) & 0 & Sen(\theta) & 0\\ 0 & 1 & 0 & 0\\ -Sen(\theta) & 0 & Cos(\theta)& 0\\ 0 & 0 & 0 & 0 \end{bmatrix}" /></a>

### Rotacion en eje z (base)
<a href="https://www.codecogs.com/eqnedit.php?latex=Rot_z=\begin{bmatrix}&space;Cos(\theta)&space;&&space;-Sen(\theta)&space;&&space;0&space;&&space;0\\&space;Sen(\theta)&space;&&space;Cos(\theta)&space;&&space;0&space;&&space;0\\&space;0&space;&&space;0&space;&&space;1&&space;0\\&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Rot_z=\begin{bmatrix}&space;Cos(\theta)&space;&&space;-Sen(\theta)&space;&&space;0&space;&&space;0\\&space;Sen(\theta)&space;&&space;Cos(\theta)&space;&&space;0&space;&&space;0\\&space;0&space;&&space;0&space;&&space;1&&space;0\\&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;\end{bmatrix}" title="Rot_z=\begin{bmatrix} Cos(\theta) & -Sen(\theta) & 0 & 0\\ Sen(\theta) & Cos(\theta) & 0 & 0\\ 0 & 0 & 1& 0\\ 0 & 0 & 0 & 0 \end{bmatrix}" /></a>


### Traslación final
<a href="https://www.codecogs.com/eqnedit.php?latex=^{i-1}T_i=T_{traslacion}(Joint_i)&plus;Rot(\theta_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?^{i-1}T_i=T_{traslacion}(Joint_i)&plus;Rot(\theta_i)" title="^{i-1}T_i=T_{traslacion}(Joint_i)+Rot(\theta_i)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=T_{base}=\prod_{i=1}^{n}^{i-1}T_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_{base}=\prod_{i=1}^{n}^{i-1}T_i" title="T_{base}=\prod_{i=1}^{n}^{i-1}T_i" /></a>
