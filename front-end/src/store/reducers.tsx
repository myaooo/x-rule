import { combineReducers } from 'redux';
import { ModelState, DataBaseState, SelectedDataType, FeatureState } from './state';

import {
  ActionType,
  RequestModelAction,
  ReceiveModelAction,
  RequestDatasetAction,
  ReceiveDatasetAction,
  SelectDatasetAction,
  SelectFeatureAction
  // Actions,
} from './actions';

export const initialModelState: ModelState = {
  model: null,
  isFetching: false
};

// export const initialModelBaseState: ModelBaseState = {};

export const initialDataBaseState: DataBaseState = {};

export const initialFeatureState: FeatureState = { idx: 0, count: 0 };

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

function selectFeatureReducer(state: FeatureState = initialFeatureState, action: SelectFeatureAction): FeatureState {
  switch (action.type) {
    case ActionType.SELECT_FEATURE:
      if (action.idx === state.idx) {
        const count = state.count + (action.deselect ? -1 : 1);
        if (count < 0) {
          console.error(`State count of select feature ${state.idx} is negative!`);  // tslint:disable-line
        }
        return {...state, count};
      } else {
        if (action.deselect) {
          console.log(`Deselecting a different feature ${action.idx}!`);  // tslint:disable-line
          return state;
        }
        if (state.count === 2) {
          console.log(`Feature ${state.idx} is already selected with count 2, deselect first!`);  // tslint:disable-line
          return state;
        }
        return {...state, idx: action.idx, count: 1};
      }
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
  selectedFeature: selectFeatureReducer
});
