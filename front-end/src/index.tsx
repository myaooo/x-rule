import * as React from 'react';
import * as ReactDOM from 'react-dom';
import {
  BrowserRouter as Router,
  Route,
} from 'react-router-dom';
import { Provider } from 'react-redux';
import { createStore } from './store';
import App from './containers/App';
import registerServiceWorker from './registerServiceWorker';
import './index.css';

const store = createStore();

ReactDOM.render(
  <Provider store={store}>
    <Router>
      <Route path="/:modelName" component={App}/>
    </Router>
  </Provider>,
  document.getElementById('root') as HTMLElement
);
registerServiceWorker();
