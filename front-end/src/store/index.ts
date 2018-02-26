import { createStore as reduxCreateStore, applyMiddleware, Store } from 'redux';
import thunk from 'redux-thunk';
import { createLogger } from 'redux-logger';
import { rootReducer } from './reducers';
import { RootState } from './state';
export * from './actions';
export * from './reducers';
export * from './state';
export * from './selectors';

export const createStore = 
  (): Store<RootState> => reduxCreateStore<RootState>(rootReducer, applyMiddleware(thunk, createLogger()));

// export Dispatch;

// export default {
//   store,
//   getModel,
// };