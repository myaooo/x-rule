import * as React from 'react';
import * as d3 from 'd3';

import { RuleList, DataSet, Streams, ConditionalStreams } from '../../models';
// import * as nt from '../../service/num';

import './index.css';
import { labelColor as defaultLabelColor, ColorType } from '../Painters/Painter';
import RuleMatrixPainter from './Painter';

interface RuleMatrixPropsOptional {
  transform: string;
  rectWidth: number;
  rectHeight: number;
  minSupport: number;
  intervalY: number;
  intervalX: number;
  flowWidth: number;
  labelColor: ColorType;
}

export interface RuleMatrixProps extends Partial<RuleMatrixPropsOptional> {
  model: RuleList;
  support: number[][] | number[][][];
  dataset?: DataSet;
  streams?: Streams | ConditionalStreams;
}

export interface RuleMatrixState {
  painter: RuleMatrixPainter;
}

export default class RuleMatrix extends React.PureComponent<RuleMatrixProps, RuleMatrixState> {
  public static defaultProps: Partial<RuleMatrixProps> & RuleMatrixPropsOptional = {
    transform: '',
    rectWidth: 30,
    rectHeight: 30,
    minSupport: 0.01,
    intervalY: 10,
    intervalX: 0.2,
    flowWidth: 60,
    labelColor: defaultLabelColor,
  };
  // private stateUpdated: boolean;
  private ref: SVGGElement;
  // private painter: RuleMatrixPainter;

  constructor(props: RuleMatrixProps) {
    super(props);
    // this.stateUpdated = false;
    const painter = new RuleMatrixPainter();
    this.state = {painter};
  }

  // componentWillReceiveProps(nextProps: RuleMatrixProps) {
  //   const model = nextProps.model;
  //   const newSupport = model.getSupportOrSupportMat();
    // console.log('update matrix?'); // tslint:disable-line
    // if (newSupport !== this.state.support) {
    //   this.setState({support: newSupport});
    // }
  // }

  componentDidUpdate() {
    // this.stateUpdated = false;
    this.painterUpdate();
  }
  componentDidMount() {
    // if (!this.props.react) {
    this.painterUpdate();
    // }
  }

  painterUpdate() {
    const {dataset, streams, model, rectWidth, rectHeight, flowWidth, minSupport, support} = this.props;
    console.log('updating matrix'); // tslint:disable-line
    this.state.painter.update({
      dataset,
      streams, 
      support,
      transform: `translate(100, 160)`,
      elemWidth: rectWidth,
      elemHeight: rectHeight,
      flowWidth,
      model,
      minSupport,
    })
      .render(d3.select<SVGGElement, {}>(this.ref));
  }
  render() {
    return <g ref={(ref) => ref && (this.ref = ref)}/>;
  }

}
