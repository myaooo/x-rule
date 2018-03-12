// Action Types
import { Dispatch as ReduxDispatch, Action } from 'redux';
import { ThunkAction } from 'redux-thunk';
import { RootState, TreeStyles, RuleStyles, Settings } from './state';
import {
  ModelBase,
  PlainData,
  DataTypeX,
  ConditionalStreams,
  Streams,
  createStreams,
  createConditionalStreams,
  SupportType,
} from '../models';

import dataService from '../service/dataService';
import { isConditional } from './selectors';
import { FilterType } from '../components/DataFilter';

export type Dispatch = ReduxDispatch<RootState>;

export enum ActionType {
  REQUEST_MODEL = 'REQUEST_MODEL',
  RECEIVE_MODEL = 'RECEIVE_MODEL',
  REQUEST_SUPPORT = 'REQUEST_SUPPORT',
  RECEIVE_SUPPORT = 'RECEIVE_SUPPORT',
  REQUEST_DATASET = 'REQUEST_DATASET',
  RECEIVE_DATASET = 'RECEIVE_DATASET',
  REQUEST_STREAM = 'REQUEST_STREAM',
  RECEIVE_STREAM = 'RECEIVE_STREAM',
  SELECT_DATASET = 'SELECT_DATASET',
  SELECT_FEATURE = 'SELECT_FEATURE',
  CHANGE_TREE_STYLES = 'CHANGE_TREE_STYLES',
  CHANGE_RULE_STYLES = 'CHANGE_RULE_STYLES',
  CHANGE_SETTINGS = 'CHANGE_SETTINGS',
  CHANGE_FILTERS = 'CHANGE_FILTERS',
}

export interface TypedAction<T> extends Action {
  readonly type: T;
}

export interface RequestModelAction extends TypedAction<ActionType.REQUEST_MODEL> {
  readonly modelName: string;
}

export interface ReceiveModelAction extends TypedAction<ActionType.RECEIVE_MODEL> {
  readonly model: ModelBase | null;
}

export interface RequestSupportAction extends TypedAction<ActionType.REQUEST_SUPPORT> {
  readonly modelName: string;
  readonly data: DataTypeX;
}

export interface ReceiveSupportAction extends TypedAction<ActionType.RECEIVE_SUPPORT> {
  readonly support: SupportType;
  readonly modelName: string;
}

export interface RequestDatasetAction extends TypedAction<ActionType.REQUEST_DATASET> {
  readonly modelName: string;
  readonly dataType: DataTypeX;
}

export interface ReceiveDatasetAction extends TypedAction<ActionType.RECEIVE_DATASET> {
  readonly modelName: string;
  readonly data: PlainData;
  readonly dataType: DataTypeX;
}

interface StreamPayload {
  modelName: string;
  dataType: DataTypeX;
  conditional: boolean;
}

export interface RequestStreamAction extends TypedAction<ActionType.REQUEST_STREAM>, Readonly<StreamPayload> {}

export interface ReceiveStreamAction extends TypedAction<ActionType.RECEIVE_STREAM>, Readonly<StreamPayload> {
  readonly streams: Streams | ConditionalStreams;
}

export interface SelectDatasetAction extends TypedAction<ActionType.SELECT_DATASET> {
  readonly dataNames: DataTypeX[];
}

export interface SelectFeatureAction extends TypedAction<ActionType.SELECT_FEATURE> {
  readonly deselect: boolean;
  readonly idx: number;
}

export interface ChangeTreeStylesAction extends TypedAction<ActionType.CHANGE_TREE_STYLES> {
  readonly newStyles: Partial<TreeStyles>;
}

export interface ChangeRuleStylesAction extends TypedAction<ActionType.CHANGE_RULE_STYLES> {
  readonly newStyles: Partial<RuleStyles>;
}

export interface ChangeSettingsAction extends TypedAction<ActionType.CHANGE_SETTINGS> {
  readonly newSettings: Partial<Settings>;
}

export interface ChangeFiltersAction extends TypedAction<ActionType.CHANGE_FILTERS> {
  readonly newFilters: FilterType[];
}

export function requestModel(modelName: string): RequestModelAction {
  return {
    type: ActionType.REQUEST_MODEL,
    modelName
  };
}

export function receiveModel(model: ModelBase | null): ReceiveModelAction {
  return {
    type: ActionType.RECEIVE_MODEL,
    model
  };
}

export interface ReceiveSupportPayload {
  modelName: string;
  support: SupportType;
}

export function requestSupport({ modelName, data }: { modelName: string; data: DataTypeX }): RequestSupportAction {
  return {
    type: ActionType.REQUEST_SUPPORT,
    modelName,
    data
  };
}

export function receiveSupport({ modelName, support }: ReceiveSupportPayload): ReceiveSupportAction {
  return {
    type: ActionType.RECEIVE_SUPPORT,
    modelName,
    support
  };
}

export function requestDataset({
  modelName,
  dataType
}: {
  modelName: string;
  dataType: DataTypeX;
}): RequestDatasetAction {
  return {
    type: ActionType.REQUEST_DATASET,
    modelName,
    dataType
  };
}

export interface ReceiveDatasetPayload {
  modelName: string;
  data: PlainData;
  dataType: DataTypeX;
}

export function receiveDataset(payload: ReceiveDatasetPayload): ReceiveDatasetAction {
  return {
    type: ActionType.RECEIVE_DATASET,
    ...payload
  };
}

export function requestStream(payload: StreamPayload): RequestStreamAction {
  return {
    type: ActionType.REQUEST_STREAM,
    ...payload
  };
}

export function receiveStream(payload: StreamPayload & { streams: Streams | ConditionalStreams }): ReceiveStreamAction {
  return {
    type: ActionType.RECEIVE_STREAM,
    ...payload
  };
}

export function selectDataset(dataNames: DataTypeX[]): SelectDatasetAction {
  return {
    type: ActionType.SELECT_DATASET,
    dataNames
  };
}

export function selectFeature({ idx, deselect }: { idx: number; deselect: boolean }): SelectFeatureAction {
  return {
    type: ActionType.SELECT_FEATURE,
    deselect,
    idx
  };
}

export function changeTreeStyles(newStyles: Partial<TreeStyles>): ChangeTreeStylesAction {
  return {
    type: ActionType.CHANGE_TREE_STYLES,
    newStyles
  };
}

export function changeRuleStyles(newStyles: Partial<RuleStyles>): ChangeRuleStylesAction {
  return {
    type: ActionType.CHANGE_RULE_STYLES,
    newStyles
  };
}

export function changeSettings(newSettings: Partial<Settings>): ChangeSettingsAction {
  return {
    type: ActionType.CHANGE_SETTINGS,
    newSettings
  };
}

export function changeFilters(newFilters: FilterType[]): ChangeFiltersAction {
  return {
    type: ActionType.CHANGE_FILTERS,
    newFilters
  };
}

export type AsyncAction = ThunkAction<any, RootState, {}>;

function fetchDataWrapper<ArgType, ReturnType>(
  fetchFn: (arg: ArgType, getState: () => RootState) => Promise<ReturnType>,
  requestAction: (arg: ArgType) => Action,
  receiveAction: (ret: ReturnType | null) => Action,
  needFetch: (arg: ArgType, getState: () => RootState) => boolean
): ((arg: ArgType) => AsyncAction) {
  // ): ThunkAction<any, RootState, ArgType> {
  const fetch = (fetchArg: ArgType, getState: () => RootState): Dispatch => {
    return (dispatch: Dispatch): Promise<Action> => {
      dispatch(requestAction(fetchArg));
      return fetchFn(fetchArg, getState).then((returnData: ReturnType | undefined) => {
        if (returnData) return dispatch(receiveAction(returnData));
        return dispatch(receiveAction(null));
      });
    };
  };
  return (arg: ArgType) => {
    return (dispatch: Dispatch, getState: () => RootState) => {
      if (needFetch(arg, getState)) {
        return dispatch(fetch(arg, getState));
      }
      return;
    };
  };
}

export const fetchModelIfNeeded = fetchDataWrapper(
  dataService.getModel,
  requestModel,
  receiveModel,
  (modelName: string, getState: () => RootState) => {
    const modelState = getState().model;
    return modelState.model === null || (!modelState.isFetching);
  }
);

// export const fetchSupportIfNeeded = fetchDataWrapper(
//   data
// );

type DatasetArg = { modelName: string; dataType: DataTypeX };

function fetchModelData({ modelName, dataType }: DatasetArg): Promise<ReceiveDatasetPayload> {
  return dataService.getModelData(modelName, dataType).then(data => ({
    data,
    modelName,
    dataType
  }));
}

export const fetchDatasetIfNeeded = fetchDataWrapper(
  fetchModelData,
  requestDataset,
  receiveDataset,
  ({ modelName, dataType }: DatasetArg, getState: () => RootState): boolean => {
    return !(dataType in getState().dataBase);
  }
);

type FetchSupportArg = {modelName: string; data: DataTypeX};

function fetchSupport(
  { modelName, data }: FetchSupportArg, getState: () => RootState
): Promise<ReceiveSupportPayload> {
  const fetcher: (modelName: string, data: DataTypeX) => Promise<SupportType> 
    = getState().settings.supportMat ? dataService.getSupportMat : dataService.getSupport;
  return fetcher(modelName, data)
    .then((support: SupportType) => ({
      support,
      modelName
    }));
}

export const fetchSupportIfNeeded = fetchDataWrapper(
  fetchSupport,
  requestSupport,
  receiveSupport,
  () => true
);

export const fetchStreamIfNeeded = fetchDataWrapper(
  (payload: StreamPayload): Promise<StreamPayload & { streams: Streams | ConditionalStreams }> => {
    const { modelName, dataType, conditional } = payload;
    return dataService.getStream(modelName, dataType, conditional).then(data => {
      if (conditional)
        return {
          streams: createConditionalStreams(data as ConditionalStreams),
          ...payload
        };
      return {
        streams: createStreams(data as Streams),
        ...payload
      };
    });
  },
  requestStream,
  receiveStream,
  (payload: StreamPayload, getState: () => RootState): boolean => {
    const streamBase = getState().streamBase[payload.dataType];
    if (!streamBase) return true;
    if (payload.conditional) return !streamBase.conditionalStreams;
    return !streamBase.streams;
  }
);

export function selectDatasetAndFetchSupport(dataNames: DataTypeX[]): ThunkAction<void, RootState, {}> {
  return (dispatch: Dispatch, getState: () => RootState) => {
    const state = getState();
    const modelState = state.model;
    const model = modelState.model;
    if (model === null || modelState.isFetching) return;
    const modelName = model.name;
    const conditional = isConditional(state);

    // only fetch support for the first data (focus)
    if (dataNames.length > 0) {
      console.log('Fetching support'); // tslint:disable-line
      dispatch(fetchSupportIfNeeded({ modelName, data: dataNames[0] }));
      dispatch(fetchStreamIfNeeded({modelName, dataType: dataNames[0], conditional}));
    }
    dispatch(selectDataset(dataNames));
  };
}

export function changeSettingsAndFetchData(newSettings: Partial<Settings>): ThunkAction<void, RootState, {}> {
  return (dispatch: Dispatch, getState: () => RootState) => {
    const state = getState();
    const modelState = state.model;
    const dataNames = state.selectedData;
    const model = modelState.model;
    if (model === null || modelState.isFetching) return;
    const modelName = model.name;
    const conditional = newSettings.conditional === undefined ? isConditional(state) : newSettings.conditional;

    // only fetch support for the first data (focus)
    if (dataNames.length > 0) {
      console.log('Fetching stream'); // tslint:disable-line
      dispatch(fetchSupportIfNeeded({ modelName, data: dataNames[0] }));
      dispatch(fetchStreamIfNeeded({modelName, dataType: dataNames[0], conditional}));
    }
    dispatch(changeSettings(newSettings));
  };
}

// export function changeFilterAndFetchData

// export const fetchDatasetAndSelect = (datasetName,)

export type Actions =
  | RequestModelAction
  | ReceiveModelAction
  | RequestDatasetAction
  | ReceiveDatasetAction
  | SelectFeatureAction
  | SelectDatasetAction
  | ChangeTreeStylesAction
  | ChangeRuleStylesAction;
