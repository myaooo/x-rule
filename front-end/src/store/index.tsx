import { createStore as reduxCreateStore, applyMiddleware } from 'redux';
import thunk from 'redux-thunk';
import { createLogger } from 'redux-logger';
import { rootReducer } from './reducers';
export * from './actions';
export * from './reducers';
export * from './state';
export * from './selectors';

export const createStore = () => reduxCreateStore(rootReducer, applyMiddleware(thunk, createLogger()));

// export Dispatch;

// export default {
//   store,
//   getModel,
// };