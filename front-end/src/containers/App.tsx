import * as React from 'react';
import { Layout, Row, Col } from 'antd';
import './App.css';
import ModelView from './ModelView';
import SideBar from './SideBar';
import DataInput from './DataInput';

const { Sider } = Layout;

export interface AppProps {
  match: { params: { modelName: string } };
}

export interface AppState {
  collapsed: boolean;
}

class App extends React.Component<AppProps, AppState> {
  constructor(props: AppProps) {
    super(props);
    this.state = {
      collapsed: false,
    };
    this.onCollapse = this.onCollapse.bind(this);
  }
  onCollapse(collapsed: boolean) {
    // console.log(collapsed);
    this.setState({ collapsed });
  }
  render() {
    const { match } = this.props;
    return (
      <div className="App">
        <Layout style={{ minHeight: '100vh' }}>
          <Sider
            collapsible={true}
            collapsed={this.state.collapsed}
            onCollapse={this.onCollapse}
            width={250}
            collapsedWidth={80}
          >
            <SideBar collapsed={this.state.collapsed}/>
          </Sider>
          <Col>
            <Row>
              <DataInput/>
            </Row>
            {/* <h1>Rule Inspector </h1> */}
            {/* <Row gutter={16} type="flex" justify="space-around"> */}
              {/* <Col span={1}/> */}
              {/* <Col span={6}> */}
                
              {/* </Col> */}
              {/* <Col span={24}> */}
            <Row>
              <ModelView modelName={match.params.modelName} />
            </Row>
          </Col>
            {/* </Col> */}
              {/* <Col span={1}/> */}
          {/* </Row> */}
        </Layout>
      </div>
    );
  }
}

export default App;
