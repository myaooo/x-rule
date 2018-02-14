import axios from 'axios';
import { ModelBase, PlainData } from '../models';

const rootUrl =
  process.env.NODE_ENV === 'development'
    ? 'http://localhost:5000'
    : location.origin;
const api = `${rootUrl}/api`;

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

function getData(data: string, isTrain: boolean = true): Promise<PlainData> {
  const url = `${api}/data/${data}`;
  const params = {isTrain};
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

// function getModel(model: string) {
//     const url = `${api}/model`;
//     axios.get(url, {
//         params: {
//             name: model,
//         }})
//     .then((response) => {
//         if (response.status === 200)
//             return response.data;
//         throw response;
//     }).catch((response) => {
//         throw response;
//     });
// }

export default {
  getModel,
  getData,
};
