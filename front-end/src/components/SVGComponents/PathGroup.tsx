import * as React from 'react';
import { NodeGroup } from 'react-move';

type PathState = {d: string};
class PathNodeGroup extends NodeGroup<any, PathState> {}

export interface PathGroupProps<T> {
  data: T[];
  d: (d?: T, i?: number) => string;
  strokeWidth: (d: T, i: number) => number;
  stroke: (d: T, i: number) => string;
  transform?: string;
  className?: string;
  animation?: boolean;
}

export interface PathGroupState {
}

// Render a group of rects
export default class PathGroup<T> extends React.PureComponent<PathGroupProps<T>, PathGroupState> {
  constructor(props: PathGroupProps<T>) {
    super(props);
  }

  render() {
    const {d, stroke, data, strokeWidth, animation, ...rest} = this.props;
    const getTargetState = (datum: T, i: number) => (
      {d: [d(datum, i)]}
    );
    if (animation)
      return (
        <PathNodeGroup 
          data={data} 
          keyAccessor={(datum, i) => i.toString()}
          start={(datum, i) => ({d: d()})}
          enter={getTargetState}
          update={getTargetState}
          leave={(datum, i) => ({d: [d()]})}
        >
          {(nodes) => (
            <g {...rest}>
              {nodes.map(({key, state, ...p}) => {
                // const { x, y, height, width } = state;
                const i = Number(key);
                return (
                  <path key={key} strokeWidth={strokeWidth(p.data, i)} stroke={stroke(p.data, i)} {...state}/>
                );
              })}
            </g>
          )}
        </PathNodeGroup>
      );
    return (
      <g {...rest}>
        {data.map((datum: T, i: number) => {
          // const { x, y, height, width } = state;
          return (
            <path key={i} d={d(datum, i)} strokeWidth={strokeWidth(datum, i)} stroke={stroke(datum, i)} />
          );
        })}
      </g>
    );
  }
}
