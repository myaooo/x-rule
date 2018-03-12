import * as React from 'react';
// import { NodeGroup } from 'react-move';
import { ColorType, defaultDuration, labelColor } from '../Painters';
// import { Rule } from '../../models';
import * as nt from '../../service/num';
import './VerticalFlow.css';
import RectGroup from './RectGroup';
import PathGroup from './PathGroup';

type Point = {x: number, y: number};
const originPoint = {x: 0, y: 0};

const curve = (s: Point = originPoint, t: Point = originPoint): string => {
  let dy = t.y - s.y;
  let dx = t.x - s.x;
  const r = Math.min(Math.abs(dx), Math.abs(dy));
  if (Math.abs(dx) > Math.abs(dy))
    return `M${s.x},${s.y} A${r},${r} 0 0 0 ${s.x + r} ${t.y} H ${t.x}`;
  else
    return `M ${s.x},${s.y} V ${s.y - r} A${r},${r} 0 0 0 ${t.x} ${t.y} `;
};

const flowCurve = (d?: {s: Point, t: Point}): string => {
  if (d) return curve(d.s, d.t);
  return curve();
};

class PathGroupT extends PathGroup<{s: Point, t: Point, width: number}> {}

interface OptionalProps {
  width: number;
  dx: number;
  dy: number;
  height: number;
  duration: number;
  color: ColorType;
  transform: string;
}

export interface VerticalFlowProps extends Partial<OptionalProps> {
  supports: number[][];
  ys: number[];  // middle point y of the target of each flow
}

export interface VerticalFlowState {
  nClasses: number;
  flows: number[][];
  reserves: number[][];
  reserveSums: number[];
  // multiplier: number;
}

export default class VerticalFlow extends React.PureComponent<VerticalFlowProps, VerticalFlowState> {
  public static defaultProps: OptionalProps = {
    width: 100,
    height: 50,
    duration: defaultDuration,
    dy: 50,
    dx: 50,
    color: labelColor,
    transform: '',
  };

  public static computeState(props: VerticalFlowProps): VerticalFlowState {
    const newProps = {...(VerticalFlow.defaultProps), ...props};
    const {supports} = newProps;
    const flows = supports;
    const nClasses = supports[0].length;
    let reserves: number[][] = 
      Array.from({length: nClasses}, (_, i) => supports.map(support => support[i]));
    reserves = reserves.map(reserve => nt.cumsum(reserve.reverse()).reverse());
    const reserveSums = new Array(supports.length).fill(0);
    reserves.forEach(reserve => nt.add(reserveSums, reserve, false));

    // const multiplier = width / reserveSums[0];
    reserves = Array.from({length: flows.length}, (_, i) => reserves.map(reserve => reserve[i]));
    return {
      flows,
      nClasses,
      reserves,
      reserveSums,
      // multiplier,
    };
  }
  
  constructor(props: VerticalFlowProps) {
    super(props);
    this.state = VerticalFlow.computeState(props);
  }

  componentWillReceiveProps(nextProps: VerticalFlowProps) {
    if (nextProps.supports === this.props.supports) return;
    this.setState(VerticalFlow.computeState(nextProps));
  }
  render() {
    const {dx, dy, ys, width, height, transform, color} = this.props as OptionalProps & VerticalFlowProps;
    const {flows, reserves, reserveSums} = this.state;
    const multiplier = width / reserveSums[0];
    // console.log(this.state); // tslint:disable-line
    // console.log(ys); // tslint:disable-line
    const heights = ys.map((y, i) => i > 0 ? y - ys[i - 1] : height);
    return (
      <g transform={transform}>
        <g className="v-reserves" transform={`translate(${-width}, 0)`}>
          {ys.map((_y, i: number) => {
            // console.log(i); // tslint:disable-line 
            const y = _y - heights[i] - dy;
            const widths = reserves[i].map((r) => r * multiplier);
            const hs = new Array(widths.length).fill(heights[i]);
            const xs = [0, ...(nt.cumsum(widths.slice(0, -1)))];
            return (
              <RectGroup
                key={i}
                widths={widths}
                xs={xs}
                heights={hs}
                transform={`translate(0,${y})`}
                fill={color}
              />
            );
          })}
        </g>
        <g className="v-flows" transform={`translate(${-width}, 0)`}>
          {ys.map((y, k: number) => {
            // const y = _y - heights[k] - dy;
            let x0 = ((k === reserves.length - 1) ? 0 : reserveSums[k + 1]) * multiplier;
            let y1 = nt.sum(flows[k]) * multiplier / 2;
            const data = flows[k].map((f: number, i: number) => {
              const pathWidth = f * multiplier;
              const s = {x: x0 + pathWidth / 2, y: -dy};
              const t = {x: dx + width, y: y1 - pathWidth / 2};
              x0 += pathWidth;
              y1 -= pathWidth;
              return {s, t, width: pathWidth};
            });
            return (
              <PathGroupT
                key={k}
                data={data}
                d={flowCurve}
                strokeWidth={(d, i) => d.width}
                stroke={(d, i) => color(i)}
                transform={`translate(0,${y})`}
              />
            );
          })}
        </g>
      </g>
    );
    // return (
    //   <g transform={transform}>
    //     <FlowNodeGroup
    //       data={ys}
    //       keyAccessor={(d, i) => i.toString()}
    //       start={(d, i) => ({ y: d - heights[i] - dy})}
    //       enter={(d, i) => ({ y: [d - heights[i] - dy]})}
    //       update={(d, i) => ({ y: [d - heights[i] - dy]})}
    //     >
    //       {(nodes) => (
    //         <g className="v-reserves" transform={`translate(${-width}, 0)`}>
    //           {nodes.map(({key, data, state}) => {
    //             const {y} = state;
    //             const i = Number(key);
    //             // console.log(i); // tslint:disable-line 
    //             const widths = reserves[i].map((r) => r * multiplier);
    //             const hs = new Array(widths.length).fill(heights[i]);
    //             const xs = [0, ...(nt.cumsum(widths.slice(0, -1)))];
    //             return (
    //               <RectGroup
    //                 key={key}
    //                 widths={widths}
    //                 xs={xs}
    //                 heights={hs}
    //                 transform={`translate(0,${y})`}
    //                 fill={color}
    //               />
    //             );
    //           })}
    //         </g>
    //       )}
    //     </FlowNodeGroup>
    //     <FlowNodeGroup
    //       data={ys}
    //       keyAccessor={(d, i) => i.toString()}
    //       start={(d, i) => ({ y: d, height: heights[i]})}
    //       enter={(d, i) => ({ y: [d]})}
    //       update={(d, i) => ({ y: [d]})}
    //     >
    //       {(nodes) => (
    //         <g className="v-flows" transform={`translate(${-width}, 0)`}>
    //           {nodes.map(({key, state}) => {
    //             const y = state.y;
    //             const k = Number(key);
    //             let x0 = ((k === reserves.length - 1) ? 0 : reserveSums[k + 1]) * multiplier;
    //             let y1 = nt.sum(flows[k]) * multiplier / 2;
    //             const data = flows[k].map((f: number, i: number) => {
    //               const pathWidth = f * multiplier;
    //               const s = {x: x0 + pathWidth / 2, y: -dy};
    //               const t = {x: dx + width, y: y1 - pathWidth / 2};
    //               x0 += pathWidth;
    //               y1 -= pathWidth;
    //               return {s, t, width: pathWidth};
    //             });
    //             return (
    //               <PathGroupT
    //                 key={key}
    //                 data={data}
    //                 d={flowCurve}
    //                 strokeWidth={(d, i) => d.width}
    //                 stroke={(d, i) => color(i)}
    //                 transform={`translate(0,${y})`}
    //               />
    //             );
    //           })}
    //         </g>
    //       )}
    //     </FlowNodeGroup>
    //   </g>
    // );
  }
}
