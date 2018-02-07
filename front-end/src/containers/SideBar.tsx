import { Card, Divider } from 'antd';
import * as React from 'react';
import { connect } from 'react-redux';
import { RuleModel, ModelBase } from '../models';
import { RootState, getModel } from '../store';
import DataSelector from './DataSelector';
import './SideBar.css';

export interface SideBarStateProp {
  model: RuleModel | ModelBase | null;
  // modelIsFetching: boolean;
  // data: PlainData | undefined;
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
}

export interface SideBarState {
}

class SideBar extends React.Component<SideBarProps, SideBarState> {
  constructor(props: SideBarProps) {
    super(props);

  }

  render() {
    const {model} = this.props;
    let dataSelector;
    if (model === null) {
      dataSelector = (<div/>);
    } else {
      dataSelector = (<DataSelector datasetName={model.dataset}/>);
    }
    return (
      <Card>
        <Divider>Controls</Divider>
        {dataSelector}
      </Card>
    );
  }
}

export default connect(mapStateToProps)(SideBar);