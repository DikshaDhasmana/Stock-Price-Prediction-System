import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface PriceChartProps {
  historical: { date: string; price: number }[];
  predictions: { date: string; price: number }[];
}

const PriceChart = ({ historical, predictions }: PriceChartProps) => {
  const data = [...historical, ...predictions];

  return (
    <div className="mb-4">
      <h2 className="text-xl font-bold mb-2">Stock Price Trends</h2>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="price" stroke="#1e40af" strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PriceChart;
