import * as React from 'react';
import { Checkbox, Collapse, Card, Row, Col, Slider, Switch, InputNumber, Button } from 'antd';
import { DataSet } from '../models/data';
import * as nt from '../service/num';
import './DataInput.css';
import { connect } from 'react-redux';
import { RootState } from '../store/state';
import { getSelectedData } from '../store';

const CheckboxGroup = Checkbox.Group;
const Panel = Collapse.Panel;

export interface CategoricalInputProps {
  categories: string[];
  checkedList: string[];
  featureName: string;
  onChange: (checkedList: string[]) => void;
}

export interface CategoricalInputState {
  checkAll: boolean;
  indeterminate: boolean;
}

export class CategoricalInput extends React.PureComponent<CategoricalInputProps, CategoricalInputState> {
  constructor(props: CategoricalInputProps) {
    super(props);
    this.onChange = this.onChange.bind(this);
    this.onCheckAllChange = this.onCheckAllChange.bind(this);
    this.state = this.calculateState(props.checkedList);
  }
  calculateState(checkedList: string[]) {
    const categories = this.props.categories;
    return {
      indeterminate: checkedList.length > 0 && (checkedList.length < categories.length),
      checkAll: checkedList.length === categories.length,
    };
  }
  onChange(checkedList: string[]) {
    this.setState(this.calculateState(checkedList));
    this.props.onChange(checkedList);
  }
  onCheckAllChange(checked: boolean) {
    this.setState({
      indeterminate: false,
      checkAll: checked,
    });
    this.props.onChange(checked ? this.props.categories : []);
  }
  render() {
    const {categories, checkedList, featureName} = this.props;
    return (
      <div>
        <h4 style={{fontSize: 12}}>{featureName} </h4>
        <hr/>
        <div style={{ borderBottom: '1px solid #E9E9E9' }}>
          <Checkbox
            indeterminate={this.state.indeterminate}
            onChange={e => this.onCheckAllChange(e.target.checked)}
            checked={this.state.checkAll}
            style={{fontSize: 12}}
          >
            Check all
          </Checkbox>
        </div>
        <CheckboxGroup value={checkedList} onChange={this.onChange} >
          {categories.map((category: string) => (
            <Checkbox value={category} key={category} style={{fontSize: 12, marginLeft: 0}}>{category}</Checkbox>
          ))}
        </CheckboxGroup>
      </div>
    );
  }
}

export interface NumericInputProps {
  range: [number, number];
  cutPoints: number[];
  featureName: string;
  value: number | [number, number];
  onChange: (selected: number | [number, number]) => void;
}

export interface NumericInputState {
  useRange: boolean;
  value: number;
  valueRange: [number, number];
}

export class NumericInput extends React.PureComponent<NumericInputProps, NumericInputState> {
  constructor(props: NumericInputProps) {
    super(props);
    this.onChangeValue = this.onChangeValue.bind(this);
    this.handleChangeMode = this.handleChangeMode.bind(this);
    this.state = {
      useRange: false,
      value: nt.mean(props.range),
      valueRange: props.range.slice() as [number, number],
    };
  }
  onChangeValue(input: number | [number, number]) {
    this.props.onChange(input);
  }
  handleChangeMode(useRange: boolean) {
    this.setState({useRange});
  }
  componentWillReceiveProps(nextProps: NumericInputProps) {
    const {value} = nextProps;
    if (Array.isArray(value)) 
      this.setState({valueRange: value.slice() as [number, number]});
    else
      this.setState({value: value});
  }
  render() {
    const {range, cutPoints, featureName} = this.props;
    const {value, valueRange, useRange} = this.state;
    const step = Number(((range[1] - range[0]) / 100).toPrecision(1));
    const marks: {[key: number]: string} = {};
    range.forEach((r) => marks[r] = r.toPrecision(3));
    cutPoints.forEach((c) => marks[c] = c.toPrecision(3));
    const min = Math.floor(range[0] / step) * step;
    const max = Math.ceil(range[1] / step) * step;
    const commonBase = { step, min, max };
    const common = {
      onChange: this.onChangeValue,
      marks,
      vertical: true,
      style: {height: 80, marginBottom: 12, fontSize: 10},
      ...commonBase,
    };
    return (
      <div>
        <p style={{fontSize: 12}}>
          {featureName} 
          <span style={{fontSize: 10, float: 'right'}}>
            Range: <Switch size="small" checked={useRange} onChange={this.handleChangeMode} />
          </span>
        </p>
        <hr/>
        <Row style={{height: 120}}>
          <Col span={11}>
            <Slider disabled={useRange} included={false} value={value} {...common}/>
            <InputNumber
              disabled={useRange}
              size="small"
              style={{width: 60}}
              value={value}
              onChange={(v) => this.onChangeValue(Number(v))}
              precision={-Math.log10(step) + 1}
              {...commonBase}
            />
          </Col>
          <Col span={11}>
            <Slider range={true} disabled={!useRange} value={valueRange} {...common} />
          </Col>
        </Row>
        
      </div>
    );
  }
}

export interface DataInputHeaderProps {
  onClick?: () => void;
}

export function DataInputHeader(props: DataInputHeaderProps) {
  const {onClick} = props;
  return (
    <div>
      Input
      <Button onClick={onClick} type="primary" icon="upload" style={{float: 'right', fontSize: 12}} size="small">
        Predict
      </Button>
    </div>
  );
}

export interface DataInputProps {
  dataset: DataSet;
}

export interface DataInputState {
  conditionList: (string[] | number | [number, number])[];
}

export class DataInput extends React.Component<DataInputProps, DataInputState> {
  
  public static resetState(dataset: DataSet) {
    const { categories, ranges } = dataset;
    return {
      conditionList: categories.map((category, i: number) => category ? category.slice(0, 1) : nt.mean(ranges[i]))
    };
  }

  constructor(props: DataInputProps) {
    super(props);
    this.onChangeCategories = this.onChangeCategories.bind(this);
    this.onChangeNumeric = this.onChangeNumeric.bind(this);
    this.state = DataInput.resetState(props.dataset);
  }
  
  onChangeCategories(i: number, checkedList: string[]) {
    if (this.props.dataset.categories[i]) {
      const conditionList = this.state.conditionList;
      conditionList[i] = checkedList;
      this.setState({
        conditionList
      });
    } else {
      console.warn(`Feature ${i} is not categorical!`);
    }
  }

  onChangeNumeric(i: number, value: number | [number, number]) {
    if (this.props.dataset.categories[i]) {
      console.warn(`Feature ${i} is categorical!`);
    } else {
      const conditionList = this.state.conditionList;
      conditionList[i] = value;
      this.setState({
        conditionList
      });
    }
  }

  render() {
    const { dataset } = this.props;
    const { featureNames, categories, ranges, discretizers } = dataset;
    const { conditionList } = this.state;
    // const isCategorical = dataset.isCategorical;
    return (
      <Row gutter={8} className="scrolling-wrapper">
      {featureNames.map((featureName: string, i: number) => {
        const category = categories[i];
        const cutPoints = discretizers[i].cutPoints;
        return (
          <Col className="card" key={i} span={4}>
            {/* <h4>{featureName}</h4> */}
            {/* <hr/> */}
            <Card bordered={true} type="inner" bodyStyle={{padding: 8}}>
              {category &&
                <CategoricalInput 
                  featureName={featureName}
                  categories={category} 
                  checkedList={conditionList[i] as string[]} 
                  onChange={(checkedList: string[]) => this.onChangeCategories(i, checkedList)}
                />
              }
              {!category && cutPoints &&
                <NumericInput 
                  featureName={featureName}
                  range={ranges[i]}
                  cutPoints={cutPoints}
                  value={conditionList[i] as (number | [number, number])}
                  onChange={(value: number | [number, number]) => this.onChangeNumeric(i, value)}
                />
              }
            </Card>
          </Col>
        );
      })}
      </Row>
    );
  }
}

export interface DataInputViewProps {
  dataset?: DataSet;
}

export function DataInputView (props: DataInputViewProps) {
  const { dataset } = props;
  return (
    <Collapse defaultActiveKey={[]}>
      <Panel header={<DataInputHeader/>} key="1" style={{width: 1200}}>
        {dataset && <DataInput dataset={dataset}/>}
      </Panel>
    </Collapse>);
}

const mapStateToProps = (state: RootState): DataInputViewProps => {
  return {
    dataset: getSelectedData(state)[0],
  };
};

export default connect(mapStateToProps)(DataInputView);