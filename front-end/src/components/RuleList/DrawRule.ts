import * as d3 from 'd3';

import { Histogram } from '../../models';

type Selection = d3.Selection<SVGElement, any, SVGElement, any>;

const defaultColors = d3.scaleOrdinal<number, string>(d3.schemeCategory10);

export function drawCondition(selector: Selection) {
  return;
}

export function drawRule(selector: Selection): void {
  return;
}

export interface HistogramParams {
  min: number;
  max: number;
}

export interface HistogramStyles {
  width: number;
  height: number;
  colors?: d3.ScaleOrdinal<number, string>;
}

type LineData = [number, number][];

export function drawHistogram(
  selector: Selection,
  data: Histogram[],
  params: HistogramParams,
  styles: HistogramStyles
) {
  const { min, max } = params;
  const { width, height, colors } = styles;
  const colorFn = colors ? colors : defaultColors;
  const lineDataList: LineData[] = data.map((hist): LineData => {
    const lineData: LineData = hist.counts.map((count: number, i: number): [number, number] => {
      return [hist.centers[i], count];
    });
    // this.min = hist.centers[0] - interval / 2;
    // this.max = hist.centers[nBins - 1] + interval / 2;
    return [[min, 0], ...lineData, [max, 0]];
  });

  const xScale = d3
    .scaleLinear()
    .domain([min, max]) // input
    .range([1, width - 1]); // output

  const yScale = d3
    .scaleLinear()
    .domain([0, Math.max(...data.map(hist => Math.max(...hist.counts)))]) // input
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
    .style('fill', (d: LineData, i: number) => colorFn(i));
  
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
    .style('fill', (d: LineData, i: number) => colorFn(i));
  enterAreas.merge(areas)
    .attr('d', areaGenerator);
  
  areas.exit()
    .transition()
    .duration(300)
    .style('fill', 'none')
    .remove();
}
