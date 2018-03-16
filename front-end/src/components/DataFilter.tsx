import * as React from 'react';
import { Checkbox, Row, Col, Slider, Button, List } from 'antd';
// import * as nt from '../service/num';
import './DataFilter.css';
import { ModelMeta } from '../models/base';
import * as d3 from 'd3';

const CheckboxGroup = Checkbox.Group;
// const Panel = Collapse.Panel;

export interface DataInputHeaderProps {
  onClick?: () => void;
}

export function DataInputHeader(props: DataInputHeaderProps) {
  const { onClick } = props;
  return (
    <div>
      Data Filter
      <Button onClick={onClick} type="primary" icon="upload" style={{ float: 'right', fontSize: 12 }} size="small">
        Filter
      </Button>
    </div>
  );
}

export type FilterType = number[] | null;

export interface CategoricalInputProps {
  categories: string[];
  checkedList: number[] | null;
  featureName: string;
  onChange: (checkedList: number[]) => void;
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
    this.state = this.calculateState(props.checkedList || []);
  }
  calculateState(checkedList: number[]) {
    const categories = this.props.categories;
    return {
      indeterminate: checkedList.length > 0 && (checkedList.length < categories.length),
      checkAll: checkedList.length === categories.length,
    };
  }
  onChange(checkedList: number[]) {
    this.setState(this.calculateState(checkedList));
    this.props.onChange(checkedList);
  }
  onCheckAllChange(checked: boolean) {

    this.props.onChange(checked ? d3.range(this.props.categories.length) : []);
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
        <CheckboxGroup value={checkedList || []} onChange={this.onChange} >
          {categories.map((category: string, i: number) => (
            <Checkbox value={i} key={category} style={{fontSize: 12, marginLeft: 0}}>{category}</Checkbox>
          ))}
        </CheckboxGroup>
      </div>
    );
  }
}

export interface NumericInputProps {
  // range: [number, number];
  cutPoints: number[];
  featureName: string;
  value: [number, number];
  onChange: (valueRange: [number, number]) => void;
}

export interface NumericInputState {
  // useRange: boolean;
  // value: number;
  // valueRange: [number, number];
}

export class NumericInput extends React.PureComponent<NumericInputProps, NumericInputState> {
  constructor(props: NumericInputProps) {
    super(props);
    this.onChangeValue = this.onChangeValue.bind(this);
    // this.handleChangeMode = this.handleChangeMode.bind(this);
    const {cutPoints} = props;
    this.state = {
      // useRange: false,
      // value: nt.mean(props.range),
      valueRange: [cutPoints[0], cutPoints[cutPoints.length - 1]],
    };
  }
  onChangeValue(input: [number, number]) {
    this.props.onChange(input.slice(0, 2) as [number, number]);
  }
  // handleChangeMode(useRange: boolean) {
  //   this.setState({useRange});
  // }
  // componentWillReceiveProps(nextProps: NumericInputProps) {
  //   const {value} = nextProps;
  //   if (Array.isArray(value)) 
  //     this.setState({valueRange: value.slice() as [number, number]});
  //   else
  //     this.setState({value: value});
  // }
  render() {
    const {cutPoints, featureName} = this.props;
    // const {value, valueRange, useRange} = this.state;
    const r0 = cutPoints[0];
    const r1 =  cutPoints[cutPoints.length - 1];
    const step = Number(((r1 - r0) / 100).toPrecision(1));
    const style = {transform: `translate(10px, -36px) rotate(-45deg)`};
    const marks: {[key: number]: {style?: React.CSSProperties, label: string}} = {};
    // cutPoints.forEach((r) => marks[r] = {label: r.toPrecision(3), style});
    cutPoints.forEach((c) => marks[c] = {label: c.toPrecision(3), style});
    const min = Math.floor(r0 / step) * step;
    const max = Math.ceil(r1 / step) * step;
    const common = {
      onAfterChange: this.onChangeValue,
      style: {marginTop: 24, marginBottom: 0, fontSize: 9},
      step, min, max,
      marks,
    };
    return (
      <div className="card">
        <Row gutter={6}>
          <Col span={10}>
            <span style={{fontSize: 12, marginTop: 16}}>
              {featureName} 
            </span>
          </Col>
          <Col span={14}>
            <Slider range={true} defaultValue={[min, max]} {...common} />
          </Col>
        </Row>
      </div>
    );
  }
}

export interface DataFilterHeaderProps {
  onClick?: () => void;
}

export function DataFilterHeader(props: DataFilterHeaderProps) {
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

interface ListData {
  featureName: string;
  idx: number;
  categories: string[] | null;
  cutPoints: number[] | null;
}

function computeListData(meta: ModelMeta, indices?: number[]): ListData[] {
  const {categories, ranges, discretizers, featureNames} = meta;
  if (indices === undefined) indices = d3.range(featureNames.length);
  return indices.map((i: number) => {
    const cutPoints = discretizers[i].cutPoints;
    return {
      featureName: featureNames[i], categories: categories[i], 
      cutPoints: [ranges[i][0], ...(cutPoints ? cutPoints : []), ranges[i][1]],
      idx: i
    };
  });
}

export interface DataFilterProps {
  meta: ModelMeta;
  indices?: number[];
  filters: (number[] | null)[];
  onChangeFilter: (i: number, filter: number[] | null) => void;
  onSubmitFilter?: () => void;
}

export interface DataFilterState {
  listData: ListData[];
  // conditionList: (string[] | number | [number, number])[];
}

export default class DataFilter extends React.Component<DataFilterProps, DataFilterState> {
  
  // public static resetState(meta: ModelMeta) {
  //   const { categories, ranges } = meta;
  //   return {
  //     conditionList: categories.map((category, i: number) => category ? category.slice(0, 1) : nt.mean(ranges[i]))
  //   };
  // }

  constructor(props: DataFilterProps) {
    super(props);
    // this.onChangeCategories = this.onChangeCategories.bind(this);
    // this.onChangeNumeric = this.onChangeNumeric.bind(this);
    this.state = {listData: computeListData(props.meta, props.indices)};
    // this.state = DataFilter.resetState(props.meta);
  }

  componentWillReceiveProps(nextProps: DataFilterProps) {
    const {meta, indices} = nextProps;
    if (meta !== this.props.meta || indices !== this.props.indices) {
      this.setState({listData: computeListData(meta, indices)});
    }
  }

  render() {
    const { filters, onChangeFilter, onSubmitFilter } = this.props;

    return (
      <List
        header={<DataInputHeader onClick={onSubmitFilter}/>}
        className="scrolling-wrapper"
        // itemLayout="vertical"
        dataSource={this.state.listData}
        size="small"
        renderItem={(item: ListData, i: number) => (
          <List.Item key={i}>
            {item.categories &&
              <CategoricalInput 
                featureName={item.featureName}
                categories={item.categories} 
                checkedList={filters[i]} 
                onChange={(checkedList: number[]) => onChangeFilter(item.idx, checkedList)}
              />
            }
            {item.cutPoints &&
              <NumericInput 
                featureName={item.featureName}
                // range={ranges[i]}
                cutPoints={item.cutPoints}
                value={filters[i] as [number, number]}
                onChange={(valueRange: [number, number]) => onChangeFilter(item.idx, valueRange)}
              />
            }
          </List.Item>
        )}
      />);

      // {featureNames.map((featureName: string, i: number) => {
      //   const category = categories[i];
      //   const cutPoints = discretizers[i].cutPoints;
      //   return (
      //     <Col className="card" key={i} span={4}>
      //       {/* <h4>{featureName}</h4> */}
      //       {/* <hr/> */}
      //       <Card bordered={true} type="inner" bodyStyle={{padding: 8}}>
      //         {category &&
      //           <CategoricalInput 
      //             featureName={featureName}
      //             categories={category} 
      //             checkedList={conditionList[i] as string[]} 
      //             onChange={(checkedList: string[]) => this.onChangeCategories(i, checkedList)}
      //           />
      //         }
      //         {!category && cutPoints &&
      //           <NumericInput 
      //             featureName={featureName}
      //             range={ranges[i]}
      //             cutPoints={cutPoints}
      //             value={conditionList[i] as (number | [number, number])}
      //             onChange={(value: number | [number, number]) => this.onChangeNumeric(i, value)}
      //           />
      //         }
      //       </Card>
      //     </Col>
      //   );
      // })}
      // </Row>
  }
}
