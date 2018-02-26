import * as React from 'react';
import * as d3 from 'd3';
import { NodeGroup } from 'react-move';

import { Rule, Condition } from '../../models';
import * as nt from '../../service/num';
import { ColorType, labelColor as defaultLabelColor } from '../Painters/Painter';
import RowOutput from './RowOutput';
import { DataSet, Streams, Stream } from '../../models';
import StreamPlot from '../SVGComponents/StreamPlot';

type RectState = {x: number, y?: number, height: number, width: number};
class RectNodeGroup extends NodeGroup<Condition, RectState> {}
// class ConditionNodeGroup extends NodeGroup<Condition, RectState> {}

export interface RectGlyphProps {
  // x: number;
  width: number;
  height: number;
  // expand?: boolean;
  // streams?: Stream; 
  color?: ColorType;
  range?: [number, number];
  supports?: number[];
  transform?: string;
  style?: React.CSSProperties;
  onClick?: () => void;
}

export class RectGlyph extends React.PureComponent<RectGlyphProps, any> {
  // renderStream(): JSX.Element {
  //   const {width, height, range, }
  // }
  render() {
    const {width, height, range, supports, color, ...rest} = this.props;
    let supportElements;
    let rangeRect;
    if (range) {
      const x = width * range[0];
      const rangeWidth = width * (range[1] - range[0]);
      rangeRect = (<rect x={x} width={rangeWidth} height={height} className="mc-range"/>);
      if (supports && color) {
        // const totalSupports = nt.sum(supports);
        const supportCumulated = nt.cumsum(supports);
        const yScaler = d3.scaleLinear()
          .domain([0, range[1] - range[0]])
          .range([height, 0]);

        const multiplier = height / (range[1] - range[0]);
        supportElements = (
          <g className="mc-satisfied" > 
            {supports.map((support: number, i: number) => (
              <rect
                key={i} 
                x={x}
                y={yScaler(supportCumulated[i])}
                width={rangeWidth} 
                height={support * multiplier}
                fill={color(i)}
              />
            ))}
          </g>
        );
      }
    }
    return (
      <g {...rest} className="matrix-condition">
        <rect width={width} height={height} className="mc-bg"/>
        {rangeRect}
        {supportElements}
      </g>
    );
  }
}

export interface ConditionViewProps extends RectGlyphProps {
  stream?: Stream;
  interval?: [number, number];
  streamColor?: ColorType;
}

export class ConditionView extends React.PureComponent<ConditionViewProps, any> {
  render() {
    const {stream, streamColor, interval, ...rest} = this.props;
    // if (!stream) return (<RectGlyph {...rest} />);
    const { style, supports, color, range, ...streamRest } = rest;
    return (
      <React.Fragment>
        {/* <Animate show={true} start={{opacity: 0.0}} > */}
          <RectGlyph {...rest} style={stream ? {display: 'none'} : {}}/>
        {/* </Animate> */}
        {stream && interval &&
        <StreamPlot data={stream} color={streamColor} range={interval} {...streamRest}/>}
      </React.Fragment>
    );
  }
}

interface OptionalProps {
  outputWidth: number;
  height: number;
  transform: string;
  labelColor: ColorType;
}

export interface RuleRowProps extends Partial<OptionalProps> {
  rule: Rule;
  dataset?: DataSet;
  features: number[];
  feature2Idx: number[];
  streams?: Streams;
  activeFeatures?: Set<number>;
  // outputs?: number[];
  supports?: number[];
  xs: number[];
  widths: number[];
  onClick?: (feature: number) => void;
}

export interface RuleRowState {
  rowWidth: number;
}

export default class RuleRow extends React.Component<RuleRowProps, RuleRowState> {
  public static defaultProps: OptionalProps = {
    height: 40,
    outputWidth: 100,
    transform: '',
    labelColor: defaultLabelColor,
  };
  constructor(props: RuleRowProps) {
    super(props);
    const {xs, widths} = props;
    const rowWidth = xs[xs.length - 1] + widths[widths.length - 1];
    this.state = {
      rowWidth,
    };
  }
  renderOutput() {
    const {rule, outputWidth, height, labelColor} = this.props as RuleRowProps & OptionalProps;
    const outputX = this.state.rowWidth + 5;
    return (
      <RowOutput 
        outputs={rule.output}
        height={height / 4}
        outputWidth={outputWidth}
        transform={`translate(${outputX}, 0)`}
        color={labelColor}
        className="matrix-outputs"
      />
    );
  }
  componentWillReceiveProps(nextProps: RuleRowProps) {
    const {xs, widths} = nextProps;
    if (widths !== this.props.widths || xs !== this.props.xs) {
      const rowWidth = xs[xs.length - 1] + widths[widths.length - 1];
      this.setState({rowWidth});
    }
  }
  render() {
    const { feature2Idx, widths, height, transform, rule, onClick, dataset, xs, labelColor, supports, streams } 
      = this.props as OptionalProps & RuleRowProps;
    const { activeFeatures } = this.props;
    const rowWidth = this.state.rowWidth;
    // const xs = [0, ...(nt.cumsum(widths))];
    const getConditionTargetState = (c: Condition) => (
      {
        x: [xs[feature2Idx[c.feature]]], 
        width: [widths[feature2Idx[c.feature]]], 
        height: [height],
        // timing: {delay: 0, duration: duration - 100},
      });
    // const conditionFeatures = rule.conditions.map((c) => c.feature);
    const supportsScaled = (supports && dataset) ? nt.muls(supports, 1 / dataset.data.length) : undefined;

    const cat2Interval = dataset ? ((f: number, c: number): [number, number] => {
      const intervals = dataset.discretizers[f].intervals;
      const range = dataset.ranges[f];
      const w = range[1] - range[0];
      if (intervals) {
        let low = intervals[c][0];
        let high = intervals[c][1];
        low = low === null ? 0 : (low - range[0]) / w;
        high = high === null ? 1 : (high - range[0]) / w;
        return [low, high];
      }
      return [c - 0.5, c + 0.5];
    }) : undefined;
    return (
      <g transform={transform}>
        { rule.conditions[0].feature !== -1 && 
        <RectNodeGroup 
          data={rule.conditions.slice()} 
          keyAccessor={(d, i) => i.toString()}
          start={(d, i) => ({x: 0, width: 0, height: 0})}
          enter={getConditionTargetState}
          update={getConditionTargetState}
          leave={(d, i) => ({x: 0, width: 0, height: 0})}
        >
          {(nodes) => (
            <g className="matrix-rule">
              {nodes.map(({key, data, state}) => {
                const {x, ...rest} = state; 
                let range = undefined;
                if (dataset) {
                  const ratios = dataset.discretizers[data.feature].ratios;
                  const r0 = nt.sum(ratios.slice(0, data.category));
                  const r1 = r0 + ratios[data.category];
                  range = [r0, r1] as [number, number];
                }
                const trans = `translate(${x},0)`;
                const stream = (activeFeatures && activeFeatures.has(data.feature) && streams) 
                  ? streams[data.feature] : undefined;
                const interval = cat2Interval && cat2Interval(data.feature, data.category);
                return (
                  <ConditionView 
                    key={key} 
                    onClick={onClick && (() => onClick(data.feature))}
                    range={range}
                    interval={interval}
                    transform={trans}
                    supports={supportsScaled}
                    color={labelColor}
                    stream={stream}
                    {...rest}
                  />
                );
              })}
            </g>
          )}
        </RectNodeGroup>
        }
        <rect width={rowWidth} height={height} className="matrix-bg"/>
        {this.renderOutput()}
        
      </g>
    );
  }
}
