import { useEffect } from 'react';
import { useLocation, Outlet } from 'react-router-dom';
import Topbar from '../Topbar/Topbar';
import Sidebar from '../Sidebar/Sidebar';
import style from './layout.module.css';

function ScrollToHash() {
    const { hash } = useLocation();

    useEffect(() => {
        if (!hash) return;
        const el = document.querySelector(hash);
        el?.scrollIntoView({ behavior: 'smooth' });
    }, [hash]);

    return null;
}

export default function LearnLayout() {
    return (
        <>
            <ScrollToHash />
            <Topbar />
            <Sidebar />
            <div className={`${style.layout}`}>
                <Outlet />
            </div>
        </>
    ); 
}