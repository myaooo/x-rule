import * as d3 from 'd3';

// export interface Painter<DataType> {
//   // new (data: DataType, styles: StyleType, params: ParamType): Painter<DataType, StyleType, ParamType>
//   update<GElement extends d3.BaseType>(
//     selector: d3.Selection<SVGElement, DataType, GElement, any>,
//     ...args: any[]
//   ): void;
//   // doJoin<GElement extends d3.BaseType>(selector: d3.Selection<SVGElement, DataType, GElement, any>): 
//   // d3.Selection<SVGElement, DataType, GElement, any>;
//   // doEnter<GElement extends d3.BaseType>(entered: d3.Selection<d3.EnterElement, DataType, GElement, any>): 
//   // d3.Selection<SVGElement, DataType, GElement, any>;
//   // doUpdate<GElement extends d3.BaseType>(merged: d3.Selection<SVGElement, DataType, GElement, any>): void;
//   // doExit<PElement extends d3.BaseType>(exited: d3.Selection<SVGElement, DataType, PElement, any>): void;
// } 

export interface Painter<DataType, ParamsType> {
  update(
    params: ParamsType
  ): this;
  data(newData: DataType): this;
  render<GElement extends d3.BaseType>(
    selector: d3.Selection<SVGElement, DataType, GElement, any>,
  ): this;
} 

export type ColorType = (i: number) => string;

export const defaultColor: ColorType = d3.scaleOrdinal<number, string>(d3.schemeCategory10);

export const defaultDuration = 400;