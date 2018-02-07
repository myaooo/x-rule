import * as React from 'react';
import { Row, Col } from 'antd';
import './App.css';
import ModelView from './ModelView';
import SideBar from './SideBar';

export interface AppProps {
  match: { params: { modelName: string } };
}

class App extends React.Component<AppProps, {}> {
  render() {
    const { match } = this.props;
    return (
      <div className="App">
        <h1>Rule Inspector </h1>
        {/* <div className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
          <h2>Welcome to React</h2>
        </div>
        <p className="App-intro">
          To get started, edit <code>src/App.tsx</code> and save to reload.
        </p> */}
        <Row gutter={16} type="flex" justify="space-around">
          {/* <Col span={1}/> */}
          <Col span={6}>
            <SideBar/>
          </Col>
          <Col span={18}>
            <ModelView modelName={match.params.modelName} />
          </Col>
            {/* <Col span={1}/> */}
          </Row>
      </div>
    );
  }
}

export default App;
