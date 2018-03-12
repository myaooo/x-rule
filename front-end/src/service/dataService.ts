import axios from 'axios';
import { ModelBase, PlainData, DataType, SurrogateDataType } from '../models';
import { DataTypeX, Stream, BasicData } from '../models/data';
import { nBins } from '../config';

const rootUrl =
  process.env.NODE_ENV === 'development'
    ? 'http://localhost:5000'
    : location.origin;
const api = `${rootUrl}/api`;

// interface Cache<T> {
//   count: number;
//   data: T;
// }

// function memorizePromise<T>(
//   f: (...a: any[]) => Promise<T>
// ): (...a: any[]) => Promise<T> {
//   const cache: {[key: string]: Cache<T>} = {};
//   return function (...a: any[]) {
//     const key = a.map((e) => JSON.stringify(a)).join(',');
//     if (key in cache)
//       return Promise.resolve<T>(cache[key].data);
//     else
//       return f(...a).then((data) => {
//         cache[key] = {data, count: 0};
//         return data;
//       });
//   };
// }
export type Filter = number[] | null;

function getOrPost<T>(url: string, params?: any, data?: any): Promise<T> {
  if (!data) {
    const config = params ? {params} : undefined;
    return axios
      .get(url, config)
      .then(response => {
        // console.log(response);  // tslint:disable-line
        if (response.status === 200) return response.data;
        throw response;
      })
      .catch(error => {
        console.warn(error);
      });
  } else {
    return axios
      .post(url, data, { params })
      .then(response => {
        if (response.status === 200) return response.data;
        throw response;
      })
      .catch(error => {
        console.log(error);  // tslint:disable-line
      });
  }
}

export function getModelList(): Promise<ModelBase> {
  const url = `${api}/model`;
  return getOrPost(url);
}

export function getModel(model: string): Promise<ModelBase> {
  const url = `${api}/model/${model}`;
  return getOrPost(url);
}

// function getData(dataName: string, data: DataType | SurrogateDataType = 'train'): Promise<PlainData> {
//   const url = `${api}/data/${dataName}`;
//   const params = {data};
//   return axios
//     .get(url, { params })
//     .then(response => {
//       if (response.status === 200) return response.data;
//       throw response;
//     })
//     .catch(error => {
//       console.log(error);  // tslint:disable-line
//     });
// }

function getModelData(
  modelName: string, 
  data: DataType | SurrogateDataType = 'train',
  filters: Filter[] | null = null,
): Promise<PlainData> {
  const url = `${api}/model_data/${modelName}`;
  const params = {data, bins: nBins};
  return getOrPost(url, params, filters);
}

export function getFilterData(
  modelName: string, 
  data: DataType | SurrogateDataType = 'train',
  filters: Filter[] | null = null,
  start: number = 0,
  end: number = 100,
): Promise<BasicData> {
  // if (filters === null) return getModelData(modelName, data);
  const url = `${api}/query/${modelName}`;
  const params = {data, bins: nBins, start, end};
  // const post = {filters, start, end};
  return axios
    .post(url, filters, { params })
    .then(response => {
      if (response.status === 200) return response.data;
      throw response;
    })
    .catch(error => {
      console.log(error);  // tslint:disable-line
    });
}

export function getSupport(
  modelName: string, data: DataTypeX = 'train',
  filters: Filter[] | null = null,
): Promise<number[][]> {
  const url = `${api}/support/${modelName}`;
  const params = {data};
  return getOrPost(url, params, filters);
}

export function getStream(
  modelName: string, data: DataTypeX = 'train', conditional: boolean = false,
  filters: Filter[] | null = null,
): Promise<Stream[] | Stream[][]> {
  const url = `${api}/stream/${modelName}`;
  const params = {data, conditional, bins: nBins};
  return getOrPost(url, params, filters);
}

function getSupportMat(
  modelName: string, data: DataTypeX = 'train', filters: Filter[] | null = null
): Promise<number[][][]> {
  const url = `${api}/support/${modelName}`;
  const params = {data, support: 'mat'};
  return getOrPost(url, params, filters);
}

export default {
  getModelList,
  getModel,
  // getData,
  getModelData,
  getFilterData,
  getSupport,
  getSupportMat,
  getStream,
};
