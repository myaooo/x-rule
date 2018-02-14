import * as d3 from 'd3';
import { Painter, ColorType, defaultColor } from './Painter';
import { Histogram } from '../../models';

type Point = [number, number];
type Line = Point[];

export interface HistPainterParams {
  // min?: number;
  // max?: number;
  width: number;
  height: number;
  colors: ColorType;
}

export class HistPainter implements Painter<Histogram[], Partial<HistPainterParams>> {
  public static defaultParams = {
    width: 100,
    height: 50,
  };
  private params: HistPainterParams;
  private hists: Histogram[];
  public update(params: HistPainterParams) {
    this.params = {...(HistPainter.defaultParams), ...(this.params), ...params};
    return this;
  }
  public data(hists: Histogram[]) {
    this.hists = hists;
    return this;
  }
  public render(selector: d3.Selection<SVGElement, any, any, any>) {
    const { width, height, colors } = this.params;
    const hists = this.hists;
    const binSizes = hists.map((hist: Histogram) => hist.centers[1] - hist.centers[0]);
    const colorFn = colors ? colors : defaultColor;
    const lineDataList: Line[] = this.hists.map((hist): Line => {
      return hist.counts.map((count: number, i: number): [number, number] => {
        return [hist.centers[i], count];
      });
    });
    const xMin = Math.min(...(lineDataList.map((line, i) => line[0][0] - binSizes[i])));
    const xMax = Math.min(...(lineDataList.map((line, i) => line[0][0] - binSizes[i])));
  
    const xScale = d3
      .scaleLinear()
      .domain([xMin, xMax]) // input
      .range([1, width - 1]); // output
  
    const yScale = d3
      .scaleLinear()
      .domain([0, Math.max(...hists.map(hist => Math.max(...hist.counts)))]) // input
      .range([height, 0]); // output
  
    const lineGenerator = d3
      .line<[number, number]>()
      .x((d: number[]): number => xScale(d[0]))
      .y((d: number[]): number => yScale(d[1]))
      .curve(d3.curveNatural);
    // .curve(d3.curveCardinal.tension(tension));
    // const lineStrings = lineDataList.map(lineData => line(lineData));
    const areaGenerator = d3
      .area<[number, number]>()
      .x((d: number[]): number => xScale(d[0]))
      .y1((d: number[]): number => yScale(d[1]))
      .y0(yScale(0))
      .curve(d3.curveNatural);
    // const areaStrings = lineDataList.map(lineData => area(lineData));
    
    const lines = selector.selectAll('.feature-dist')
      .data(lineDataList);
    const enterLines = lines.enter()
      .append('path')
      .attr('class', 'feature-dist')
      .style('fill', (d: Line, i: number) => colorFn(i));
    
    enterLines.merge(lines)
      .attr('d', lineGenerator);
  
    lines.exit()
      .transition()
      .duration(300)
      .style('fill', 'none')
      .remove();
  
    const areas = selector.selectAll('.feature-dist-area')
      .data(lineDataList);
    
    const enterAreas = areas.enter()
      .append('path')
      .classed('feature-dist-area', true)
      .style('fill', (d: Line, i: number) => colorFn(i));
    enterAreas.merge(areas)
      .attr('d', areaGenerator);
    
    areas.exit()
      .transition()
      .duration(300)
      .style('fill', 'none')
      .remove();
    return this;
  }
}
