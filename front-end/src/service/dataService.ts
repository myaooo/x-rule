import axios from 'axios';
import { ModelBase, PlainData, DataType, SurrogateDataType } from '../models';
import { DataTypeX } from '../models/data';

const rootUrl =
  process.env.NODE_ENV === 'development'
    ? 'http://localhost:5000'
    : location.origin;
const api = `${rootUrl}/api`;

function getModelList(): Promise<ModelBase> {
  const url = `${api}/model`;
  return axios
    .get(url)
    .then(response => {
      // console.log(response);  // tslint:disable-line
      if (response.status === 200) return response.data;
      throw response;
    })
    .catch(error => {
      console.log(error);  // tslint:disable-line
    });
}

function getModel(model: string): Promise<ModelBase> {
  const url = `${api}/model/${model}`;
  return axios
    .get(url)
    .then(response => {
      // console.log(response);  // tslint:disable-line
      if (response.status === 200) return response.data;
      throw response;
    })
    .catch(error => {
      console.log(error);  // tslint:disable-line
    });
}

function getData(dataName: string, data: DataType | SurrogateDataType = 'train'): Promise<PlainData> {
  const url = `${api}/data/${dataName}`;
  const params = {data};
  return axios
    .get(url, { params })
    .then(response => {
      if (response.status === 200) return response.data;
      throw response;
    })
    .catch(error => {
      console.log(error);  // tslint:disable-line
    });
}

function getSupport(
  modelName: string, data: DataTypeX = 'train'
): Promise<number[][]> {
  const url = `${api}/support/${modelName}`;
  const params = {data};
  return axios
    .get(url, { params })
    .then(response => {
      if (response.status === 200) return response.data;
      throw response;
    })
    .catch(error => {
      console.log(error);  // tslint:disable-line
    });
}

function getStream(
  modelName: string, data: DataTypeX = 'train', conditional: boolean = false
): Promise<number[][][] | number[][][][]> {
  const url = `${api}/stream/${modelName}`;
  const params = {data, conditional};
  return axios
    .get(url, { params })
    .then(response => {
      if (response.status === 200) return response.data;
      throw response;
    })
    .catch(error => {
      console.log(error);  // tslint:disable-line
    });
}

export default {
  getModelList,
  getModel,
  getData,
  getSupport,
  getStream,
};
