export type Vector =
  | Array<number>
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array;

// export type DType = 'int8' | 'int16' | 'int32' | 'uint8' | 'uint16' | 'uint32' | 'float32' | 'float64';

// export interface Array2D<T extends Vector> {
//   size: number;
//   shape: [number, number];
//   dtype: DType;
//   data: T;
// }

// export class Matrix<T extends Vector> implements Array2D<T> {
//   public size: number;
//   public shape: [number, number];
//   public dtype: DType;
//   public data: T;
//   public get(i: number, j: number) {
//     return this.data[i * this.shape[0] + j];
//   }
//   public set(i: number, j: number, v: number) {
//     this.data[i * this.shape[0] + j] = v;
//   }
// }

export function muls<T extends Vector>(a: T, b: number, copy: boolean = true): T {
  const ret = copy ? a.slice() as T : a;
  for (let i = 0; i < ret.length; ++i)
    ret[i] *= b;
  return ret;
}

export function mul<T extends Vector>(a: T, b: T, copy: boolean = true): T {
  if (a.length !== b.length) {
    throw 'Length of a and b must be equal!';
  }
  const ret = copy ? a.slice() as T : a;
  for (let i = 0; i < ret.length; ++i)
    ret[i] *= b[i];
  return ret;
}

export function add<T extends Vector>(a: T, b: T, copy: boolean = true): T {
  if (a.length !== b.length) {
    throw 'Length of a and b must be equal!';
  }
  const ret = copy ? a.slice() as T : a;
  for (let i = 0; i < ret.length; ++i)
    ret[i] += b[i];
  return ret;
}

export function sum<T extends Vector>(arr: T): number {
  let _sum: number = 0;
  for (let i = 0; i < arr.length; ++i) {
    _sum += arr[i];
  }
  return _sum;
}

export function cumsum<T extends Vector>(a: T): T {
  // if (a instanceof nj.NdArray)
  const arr = a.slice() as T;
  for (let i = 1; i < arr.length; ++i) {
    arr[i] += arr[i - 1];
  }
  return arr;
}

// export function 

// export function sum<T = number>(a: nj.NdArrayLike<T>, axis?: number): nj.NdArray<T> {
//   // if (axis === undefined) return _sum(a);
//   const dim = axis ? axis : 0;
//   const arr = nj.array<T>(a);
//   const ret = nj.zeros<T>([...(arr.shape.slice(0, dim)), ...(arr.shape.slice(dim + 1))], arr.dtype);
//   const nulls = arr.shape.map(() => null);
//   const starts  = nulls.slice(0, dim);
//   const ends = nulls.slice(dim + 1, 0);
//   for (let i = 0; i < arr.shape[dim]; i++) {
//     ret.add(arr.pick(...starts, i, ...ends), false);
//   }
//   return ret;
// }
