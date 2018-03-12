import * as React from 'react';
import { ColorType } from '../Painters/Painter';

export interface LegendProps {
  transform?: string;
  labels: string[];
  color: ColorType;
  style?: React.CSSProperties;
}

export function Legend (props: LegendProps) {
  const {labels, color, ...rest} = props;
  return (
    <g {...rest}>
      <text>False Prediction</text>
      {labels.map((label: string, i: number) => {
        return (
          <g key={label} transform={`translate(0, ${i * 30 + 20})`}> 
            <text textAnchor="start" dx="15" dy="12">{label}</text>
            <rect className="mc-stripe" width={10} height={10} fill={color(i)}/>
          </g>
        );
      })}
    </g>
  );
}
