// Action Types
import { Dispatch as ReduxDispatch, Action } from 'redux';
import { ThunkAction } from 'redux-thunk';
import { RootState, SelectedDataType } from './state';
import { ModelBase, PlainData } from '../models';

import dataService from '../service/dataService';

export type Dispatch = ReduxDispatch<RootState>;

export enum ActionType {
  REQUEST_MODEL = 'REQUEST_MODEL',
  RECEIVE_MODEL = 'RECEIVE_MODEL',
  REQUEST_DATASET = 'REQUEST_DATASET',
  RECEIVE_DATASET = 'RECEIVE_DATASET',
  SELECT_DATASET = 'SELECT_DATASET',
  SELECT_FEATURE = 'SELECT_FEATURE',
}

export interface RequestModelAction extends Action {
  readonly type: ActionType.REQUEST_MODEL;
  readonly modelName: string;
}

export interface ReceiveModelAction extends Action {
  readonly type: ActionType.RECEIVE_MODEL;
  readonly model: ModelBase;
}

export interface RequestDatasetAction extends Action {
  readonly type: ActionType.REQUEST_DATASET;
  readonly datasetName: string;
  readonly isTrain: boolean;
}

export interface ReceiveDatasetAction extends Action {
  readonly type: ActionType.RECEIVE_DATASET;
  readonly datasetName: string;
  readonly data: PlainData;
  readonly isTrain: boolean;
}

export interface SelectDatasetAction extends Action {
  readonly type: ActionType.SELECT_DATASET;
  readonly dataNames: SelectedDataType[];
}

export interface SelectFeatureAction extends Action {
  readonly type: ActionType.SELECT_FEATURE;
  readonly deselect: boolean;
  readonly idx: number;
}

export function requestModel(modelName: string): RequestModelAction {
  return {
    type: ActionType.REQUEST_MODEL,
    modelName
  };
}

export function receiveModel(model: ModelBase): ReceiveModelAction {
  return {
    type: ActionType.RECEIVE_MODEL,
    model
  };
}

export function requestDataset({
  datasetName,
  isTrain
}: {
  datasetName: string;
  isTrain: boolean;
}): RequestDatasetAction {
  return {
    type: ActionType.REQUEST_DATASET,
    datasetName,
    isTrain
  };
}

export function receiveDataset({
  datasetName,
  data,
  isTrain
}: {
  datasetName: string;
  data: PlainData;
  isTrain: boolean;
}): ReceiveDatasetAction {
  return {
    type: ActionType.RECEIVE_DATASET,
    datasetName,
    data,
    isTrain
  };
}

export function selectDataset(dataNames: SelectedDataType[]): SelectDatasetAction {
  return {
    type: ActionType.SELECT_DATASET,
    dataNames
  };
}

export function selectFeature({idx, deselect}: {idx: number, deselect: boolean}): SelectFeatureAction {
  return {
    type: ActionType.SELECT_FEATURE,
    deselect,
    idx
  };
}

type AsyncAction = ThunkAction<any, RootState, {}>;

function fetchDataWrapper<ArgType, ReturnType>(
  fetchFn: (arg: ArgType) => Promise<ReturnType>,
  requestAction: (arg: ArgType) => Action,
  receiveAction: (ret: ReturnType) => Action,
  needFetch: (arg: ArgType, getState: () => RootState) => boolean
): ((arg: ArgType) => AsyncAction) {
  const fetch = (fetchArg: ArgType): Dispatch => {
    return (dispatch: Dispatch) => {
      dispatch(requestAction(fetchArg));
      return fetchFn(fetchArg).then((returnData: ReturnType) => {
        return dispatch(receiveAction(returnData));
      });
    };
  };
  return (arg: ArgType) => {
    return (dispatch: Dispatch, getState: () => RootState) => {
      if (needFetch(arg, getState)) {
        return dispatch(fetch(arg));
      }
    };
  };
}

export const fetchModelIfNeeded = fetchDataWrapper(
  dataService.getModel,
  requestModel,
  receiveModel,
  (modelName: string, getState: () => RootState) => {
    const modelState = getState().model;
    return modelState.model === null || modelState.isFetching;
  }
);

type DatasetArg = {datasetName: string, isTrain: boolean};

export const fetchDatasetIfNeeded = fetchDataWrapper(
  ({datasetName, isTrain}: DatasetArg): Promise<{ datasetName: string; data: PlainData; isTrain: boolean}> => {
    return dataService.getData(datasetName, isTrain).then(data => ({
      data,
      datasetName,
      isTrain
    }));
  },
  requestDataset,
  receiveDataset,
  ({datasetName, isTrain}: DatasetArg, getState: () => RootState): boolean => {
    return ! ((isTrain ? 'train' : 'test') in getState().dataBase);
  }
);

export type Actions = RequestModelAction | ReceiveModelAction | RequestDatasetAction | ReceiveDatasetAction;
