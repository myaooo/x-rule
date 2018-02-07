import * as React from 'react';
import { connect } from 'react-redux';
import { Radio, Col, Row } from 'antd';
import {
  Dispatch,
  SelectDatasetAction,
  fetchDatasetIfNeeded,
  selectDataset,
  getSelectedDataName,
  RootState
} from '../store';
import './DataSelector.css';

type DatasetType = 'train' | 'test';

export interface DataSelectorStateProp {
  selectedData: DatasetType | null;
}

const mapStateToProps = (state: RootState): DataSelectorStateProp => {
  return {
    selectedData: getSelectedDataName(state)
  };
};
export interface DataSelectorDispatchProp {
  selectData: (name: DatasetType) => SelectDatasetAction;
  loadData: (datasetName: string, isTrain: boolean) => Dispatch;
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: any): DataSelectorDispatchProp => {
  return {
    // loadModel: bindActionCreators(getModel, dispatch),
    selectData: (name: DatasetType): SelectDatasetAction => dispatch(selectDataset(name)),
    loadData: (datasetName: string, isTrain: boolean): Dispatch =>
      dispatch(fetchDatasetIfNeeded({ datasetName, isTrain }))
  };
};

export interface DataSelectorProps extends DataSelectorStateProp, DataSelectorDispatchProp {
  datasetName: string;
}

export interface DataSelectorState {
  // dataNames: string[];
  // name2Value: {[dataName: string]: number};
}

const RadioGroup = Radio.Group;
const RadioButton = Radio.Button;

class DataSelector extends React.Component <DataSelectorProps, DataSelectorState> {
  dataNames: DatasetType[];
  name2Value: {[dataName: string]: number};
  constructor(props: DataSelectorProps) {
    super(props);
    const dataNames = ['train', 'test'] as DatasetType[];
    const name2Value = {};
    dataNames.forEach((dataName: DatasetType, i: number) => name2Value[dataName] = i);
    this.dataNames = dataNames;
    this.name2Value = name2Value;
    this.selectData = this.selectData.bind(this);
  }
  componentDidMount() {
    this.props.loadData(this.props.datasetName, true);
    this.props.loadData(this.props.datasetName, false);
  }
  selectData(e: React.ChangeEvent<HTMLInputElement>) {
    const value = e.target.value;
    this.props.selectData(this.dataNames[value]);
  }
  render() {
    let selectedValue = undefined;
    if (this.props.selectedData !== null)
      selectedValue = this.name2Value[this.props.selectedData];
    return (
      <Row type="flex" align="middle">
        <Col span={8}>
          <span className="sidebar-label"> Dataset: </span>
        </Col>
        <Col span={16} className="sidebar-input">
        <RadioGroup onChange={this.selectData} value={selectedValue}>
          {this.dataNames.map((dataName, i) => (<RadioButton key={i} value={i}>{dataName}</RadioButton>))}
        </RadioGroup>
        </Col>
      </Row>
    );
  }
}

export default connect(mapStateToProps, mapDispatchToProps)(DataSelector);