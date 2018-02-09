import { Card, Divider, Collapse } from 'antd';
// import { Menu, Icon } from 'antd';
import * as React from 'react';
import { connect } from 'react-redux';
import { RuleModel, ModelBase } from '../models';
import { RootState, getModel } from '../store';
import DataSelector from './DataSelector';
import './SideBar.css';

const { Panel } = Collapse;

export interface SideBarStateProp {
  model: RuleModel | ModelBase | null;
}

const mapStateToProps = (state: RootState): SideBarStateProp => {
  return {
    model: getModel(state),
    // modelIsFetching: getModelIsFetching(state),
    // data: getData(state)
  };
};

export interface SideBarProps extends SideBarStateProp {
  // width: number;
  collapsed?: boolean;
}

export interface SideBarState {
  activeKey: string | string[];
}

const defaultActiveKey = ['1'];

class SideBar extends React.Component<SideBarProps, SideBarState> {
  constructor(props: SideBarProps) {
    super(props);
    this.state = {activeKey: defaultActiveKey};
    this.onChange = this.onChange.bind(this);
  }
  onChange(key: string | string[]) {
    this.setState({activeKey: key});
  }
  render() {
    const {model} = this.props;
    return (
      <Card bordered={false}>
        <Divider>Controls</Divider>
          <Collapse 
            bordered={false} 
            activeKey={this.props.collapsed === true ? undefined : this.state.activeKey} 
            onChange={this.onChange}
          >
            {model !== null && 
              <Panel header="Dataset: " key="1">
                <DataSelector key={'1'} datasetName={model.dataset}/>
              </Panel>}
          </Collapse>
      </Card>
      // <Menu theme="light" mode="inline">
      //   <Menu.Item key="1" disabled={true}>
      //     <Icon type="pie-chart" />
          // dataSelector
      //   </Menu.Item>
      //   <Menu.Item key="2" disabled={true}>
      //     <Icon type="pie-chart" />
      //     {/* {dataSelector} */}
      //   </Menu.Item>
      // </Menu>
    );
  }
}

export default connect(mapStateToProps)(SideBar);