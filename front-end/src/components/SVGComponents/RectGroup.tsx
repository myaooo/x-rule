import * as React from 'react';
import { NodeGroup } from 'react-move';

type RectState = {x?: number, y?: number, height?: number, width?: number};
class RectNodeGroup extends NodeGroup<number, RectState> {}

export interface RectGroupProps {
  xs: number[];
  ys?: number[];
  heights: number[];
  widths: number[];
  fill?: (i: number) => string;
  transform?: string;
  className?: string;
}

export interface RectGroupState {
}

// Render a group of rects
export default class RectGroup extends React.PureComponent<RectGroupProps, RectGroupState> {
  constructor(props: RectGroupProps) {
    super(props);
  }

  render() {
    const {xs, ys, widths, heights, fill, ...rest} = this.props;

    const getTargetState = ys 
      ? (d: any, i: number) => ({y: ys[i], width: [widths[i]], height: [heights[i]]})
      : (d: any, i: number) => ({width: [widths[i]], height: [heights[i]]});
    return (
        <RectNodeGroup 
          data={xs} 
          keyAccessor={(d, i) => i.toString()}
          start={(d, i) => ({width: 0, height: 0})}
          enter={getTargetState}
          update={getTargetState}
          leave={(d, i) => ({width: [0], height: [0]})}
        >
          {(nodes) => (
            <g {...rest}>
              {nodes.map(({key, data, state}) => {
                // const { x, y, height, width } = state;
                const i = Number(key);
                return (
                  <rect key={key} x={xs[i]} fill={fill && fill(Number(key))} {...state}/>
                );
              })}
            </g>
          )}
        </RectNodeGroup>
    );
  }
}
