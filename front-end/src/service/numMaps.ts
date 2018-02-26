// import * as nj from 'numjs';
// import * as BaseNdArray from 'ndarray';

// export * from 'numjs';

// export type DType = BaseNdArray.DataType;

// export function cumsum<T = number, RT = T>(a: nj.NdArrayLike<T>, axis?: number, dtype?: DType): nj.NdArray<RT> {
//   // if (a instanceof nj.NdArray)
//   const arr = nj.array<T, RT>(a, dtype).clone();
//   const dim = axis ? axis : 0;
//   if (axis === undefined) arr.reshape(-1);
//   // for (let _dim in )
//   const nulls = arr.shape.map(() => null);
//   const starts  = nulls.slice(0, dim);
//   const ends = nulls.slice(dim + 1, 0);
//   for (let i = 1; i < arr.shape[dim]; i++) {
//     arr.pick(...starts, i, ...ends).add(arr.pick(...starts, i - 1, ...ends), false);
//   }
//   return arr;
// }

// // const _sum = nj.sum;

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

// // export function flip<T>(a: nj.NdArray<T>, axis: number): nj.NdArray<T> {

// // }