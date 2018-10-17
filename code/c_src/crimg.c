#include "filtros.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
/**
 * Tipos de imagen representadas
 */
typedef enum tipo_imagen {
  GRISES=1,
  COLOR=2
} TipoImagen;

/**
 * Canales de color
 */
typedef enum canal {
  ROJO=0,
  VERDE=1,
  AZUL=2
} Canal;

/**
 * pixel es un entero sin signo de al menos 32 bits
 */
typedef unsigned int Pixel;

typedef struct imagen {
  TipoImagen tipo;
  int ancho;
  int alto;
  int valor_maximo;
  Pixel* pixels;
} Imagen;


#define MAXW 10

int clip(int i, int min, int max) {
  return  i <= min ? min : (i >= max ? max : i);
}

/*------------------------------------------------------------------------*/

void bordes(Pixel pixels_in[], int ancho, int alto, int max_val, int orient, Pixel pixels_out[]) 
{
  int i,j,i2,j2;
  int pder,pizq,pabj,parr,dx,dy;
  for (i = 0; i < alto ; ++i) {
    for (j = 0 ; j < ancho; ++j) {
	j2 = clip(j + 1, 0, ancho-1 );	
	pder = pixels_in[i*ancho+j2];
	j2 = clip(j - 1, 0, ancho-1 );	
	pizq = pixels_in[i*ancho+j2];
	i2 = clip(i - 1, 0, alto-1 );
	parr = pixels_in[i2*ancho+j];
	i2 = clip(i + 1, 0, alto-1 );
	pabj = pixels_in[i2*ancho+j];
	dx = pder-pizq;
        dy = pabj-parr;
	pixels_out[i*ancho+j]  = clip((int)(sqrt((double)(dx*dx+dy*dy))+0.5),0,max_val);
    }
  }
}

/*------------------------------------------------------------------------*/
void reemplazar_etiqueta(Imagen* pI, Pixel a, Pixel b, int maxi) {
  const int n = pI->ancho* pI->alto;  
  register int i;
  Pixel *pi = pI->pixels;
  if ((maxi > n) || (maxi == 0))
    maxi = n;
  for (i = 0; i < maxi; i++) {
    if (pi[i] == a) 
      pi[i] = b;
  }
}

/*------------------------------------------------------------------------*/
void reemplazar_etiqueta2(Imagen* pI, Pixel a, Pixel b, int maxp) {
  const int N = pI->ancho;
  const int M = pI->alto;  
  Pixel *pi = pI->pixels;
  register int i,j,k,ult_reemplazo = maxp;
  if ((maxp >= M*N) || (maxp == 0))
    maxp = M*N-1;
  i = maxp / N;
  j = maxp % N;
  k = maxp;
  while ( i >= 0 ) {
    while ( j >= 0 ) {
      if (pi[k] == a) {
	pi[k] = b;
	ult_reemplazo = k;
      }
      j--;
      k--;
    }
    j = N-1;
    i--;
    if ((ult_reemplazo - k) > pI->ancho) {
      break;
    }
  }
}

/*------------------------------------------------------------------------*/


void etiquetar(const Imagen* pG, int u, Imagen* pE) {
  const int ancho = pG->ancho;
  const int alto  = pG->alto;
  register int i,j,k;
  const Pixel *pg = pG->pixels;
  Pixel *pe = pE->pixels;
  Pixel *pen = pE->pixels - ancho; /* fila anterior */
  int L = 0; /* etiqueta */
  for (k = 0, i = 0; i < alto; ++i) {
    /* printf("%6d/%6d L=%6d\n",i,alto,L); */
    for (j = 0; j < ancho; ++j, ++k) {
      if (pg[k] > u) { /* estrictamente mayor segun letra */
	pe[k] = 0;
	continue;
      }
      int n = (i > 0) ? pen[k] : 0;
      int w = (j > 0) ? pe[k-1] : 0;
      if (n == 0) {
	if (w == 0) { /* ambos borde */
	  pe[k] = ++L; /* nueva etiqueta */
	} else { /* oeste no era borde */
	  pe[k] = w;
	}	
      } else { /* n no es borde */
	if (w == 0) {
	  pe[k] = n;
	}  else { /* etiquetas distintas */
	  pe[k] = w;
	  if (n != w) {
	    reemplazar_etiqueta2(pE,n,w,k); 
	  }
	}
      } 
    } /* j: dentro de cada fila */
  } /* i: para cada fila */
  pE->valor_maximo = L;
}

/*------------------------------------------------------------------------*/

