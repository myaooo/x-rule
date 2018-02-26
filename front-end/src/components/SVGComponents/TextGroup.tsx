import * as React from 'react';
import { NodeGroup } from 'react-move';

type Point = {x: number, y: number};

class TextNodeGroup extends NodeGroup<any, Partial<Point>> {}

export interface TextGroupProps {
  texts: string[];
  xs: number[];
  ys?: number[];
  transform?: string;
  className?: string;
  rotate?: number;
}

export default function TextGroup(props: TextGroupProps) {
  const {texts, xs, ys, rotate, ...rest} = props;
  const rotateDeg = rotate ? rotate : 0;
  const getTargetPos = (d: any, i: number) => {
    if (ys) return {x: [xs[i]], y: [ys[i]]};
    return {x: [xs[i]]} as {x: [number]};
  };
  return (
    <TextNodeGroup
      data={texts}
      keyAccessor={(d, i) => i.toString()}
      start={(d, i) => ({x: 0, y: 0})}
      enter={getTargetPos}
      update={getTargetPos}
      leave={(d, i) => ({x: 0, y: 0})}
    > 
      {(nodes) => {
        return (
          <g {...rest}>
            {nodes.map(({key, data, state}) => (
              <text key={key} transform={`translate(${state.x}) rotate(${rotateDeg})`}>
                {data}
              </text>
            ))}
          </g>
        );
      }}
    </TextNodeGroup>
  );
}
