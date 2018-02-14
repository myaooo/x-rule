import { combineReducers } from 'redux';
import { 
  ModelState, DataBaseState, SelectedDataType, FeatureState, TreeStyles, initTreeStyles, FeatureStatus 
} from './state';

import {
  ActionType,
  RequestModelAction,
  ReceiveModelAction,
  RequestDatasetAction,
  ReceiveDatasetAction,
  SelectDatasetAction,
  SelectFeatureAction,
  ChangeTreeStylesAction,
  // Actions,
} from './actions';

export const initialModelState: ModelState = {
  model: null,
  isFetching: false
};

// export const initialModelBaseState: ModelBaseState = {};

export const initialDataBaseState: DataBaseState = {};

export const initialFeaturesState: FeatureState[] = [];

function modelStateReducer(
  state: ModelState = initialModelState,
  action: RequestModelAction | ReceiveModelAction
): ModelState {
  switch (action.type) {
    case ActionType.REQUEST_MODEL:
      console.log("start Fetching...");  // tslint:disable-line
      return { ...state, isFetching: true };
    case ActionType.RECEIVE_MODEL:
      console.log("receiving model...");  // tslint:disable-line
      return {
        isFetching: false,
        model: action.model
      };
    default:
      return state;
  }
}

function dataBaseReducer(
  state: DataBaseState = initialDataBaseState,
  action: RequestDatasetAction | ReceiveDatasetAction
): DataBaseState {
  switch (action.type) {
    case ActionType.REQUEST_DATASET:
      return state;
    // const dataset = {};
    // dataset[action.isTrain ? 'train' : 'test'] = null;

    // return { ...state, ...dataset };
    case ActionType.RECEIVE_DATASET:
      const newState: DataBaseState = {};
      if (action.isTrain) {
        newState.train = action.data;
      } else {
        newState.test = action.data;
      }
      return {...state, ...newState};
    default:
      return state;
  }
}

function selectDatasetReducer(state: SelectedDataType[] = [], action: SelectDatasetAction): SelectedDataType[] {
  switch (action.type) {
    case ActionType.SELECT_DATASET:
      return action.dataNames;
    default:
      return state;
  }
}

function selectedFeaturesReducer(
  state: FeatureState[] = initialFeaturesState, 
  action: SelectFeatureAction
): FeatureState[] {
  switch (action.type) {
    case ActionType.SELECT_FEATURE:
      if (action.deselect) {
        const idx = state.findIndex((f: FeatureState) => f.idx === action.idx);
        if (idx === -1) return state;
        const feature = state[idx];
        feature.status--;
        if (feature.status < FeatureStatus.HOVER) {
          return [...(state.slice(0, idx)), ...(state.slice(idx + 1))];
        }
        return state;
      } else {
        const idx = state.findIndex((f: FeatureState) => f.idx === action.idx);
        if (idx === -1) 
          return [...state, {idx: action.idx, status: FeatureStatus.HOVER}];
        else if (state[idx].status < FeatureStatus.SELECT) {
          state[idx].status++;
        }
        return state;
      }
    default:
      return state;
  }
}

function treeStyleReducer(state: TreeStyles = initTreeStyles, action: ChangeTreeStylesAction): TreeStyles {
  switch (action.type) {
    case ActionType.CHANGE_TREE_STYLES:
      return {...state, ...action.newStyles};
    default:
      return state;
  }
}
// function selectedDataReducer(
//   state: string,
//   action:
// )

export const rootReducer = combineReducers({
  model: modelStateReducer,
  dataBase: dataBaseReducer,
  selectedData: selectDatasetReducer,
  selectedFeatures: selectedFeaturesReducer,
  treeStyles: treeStyleReducer,
});
